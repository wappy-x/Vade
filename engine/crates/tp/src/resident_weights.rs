//! Complete feature rows stay local. Finished output rows gather in rank order.
use crate::Result;
use engine_model::plan::{storage::WeightStorage,layer::Topology};

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Rows {pub start:usize,pub count:usize,pub total:usize,pub split:bool}
impl Rows {
    pub fn new(shape:&[u64],storage:WeightStorage,t:Topology)->Result<Self> {
        t.validate().map_err(|e|e.to_string())?;
        if shape.len()!=2 || shape.contains(&0) {return Err("resident projection requires a matrix".into());}
        let unit=match storage {WeightStorage::Fp8Block {block}=>block[0],_=>1};
        let total=usize::try_from(shape[0]).map_err(|_|"resident rows exceed address space")?;
        let stride=unit.checked_mul(u64::from(t.parts)).filter(|&n|n>0).ok_or("resident scale split overflow")?;
        let split=shape[0].is_multiple_of(stride);
        if !split {return Ok(Self {start:0,count:total,total,split:false});}
        let owner=engine_model::plan::tensor_split::Ownership::equal(shape,0,unit,t.parts).map_err(|e|e.to_string())?;
        let interval=owner.owner(t.rank).map_err(|e|e.to_string())?.interval.clone();
        Ok(Self {start:interval.start as usize,count:(interval.end-interval.start) as usize,total,split:t.parts!=1})
    }
}
#[cfg(any(target_os="linux",test))]
pub fn merge_argmax(values:&[f32],rows:&[Rows])->Result<u32> {
    if rows.is_empty() || values.len()!=rows.len()*3 {return Err("resident argmax extent differs".into());}
    let mut best=None;
    for (value,row) in values.chunks_exact(3).zip(rows) {
        if value.iter().any(|word|!word.is_finite()) || value[2]!=1.0 {return Err("nonfinite logits".into());}
        let local=value[1] as usize;
        if value[1]<0.0 || local as f32!=value[1] || local>=row.count || row.start.checked_add(row.count).is_none_or(|end|end>row.total) {
            return Err("resident argmax local ID differs".into());
        }
        let id=u32::try_from(row.start.checked_add(local).ok_or("resident argmax ID overflow")?).map_err(|_|"resident argmax ID overflow")?;
        let candidate=(value[0],id);best=Some(best.map_or(candidate,|current|spark_gpu_probe::next_math::better(current,candidate)));
    }Ok(best.unwrap().1)
}
#[cfg(any(target_os="linux",test))]
fn gathered_batch(values:&[f32],local_rows:usize,total_rows:usize,jobs:usize)->Result<Vec<Vec<f32>>> {
    let chunks=total_rows.checked_div(local_rows).filter(|&chunks|chunks>0).ok_or("resident batch row geometry is empty")?;
    let local=local_rows.checked_mul(jobs).ok_or("resident batch local extent overflow")?;
    if jobs==0 || local_rows.checked_mul(chunks)!=Some(total_rows) || local.checked_mul(chunks)!=Some(values.len()) {
        return Err("resident batch rank-major extent differs".into());
    }
    let mut output=Vec::with_capacity(jobs);
    for job in 0..jobs {let mut matrix=Vec::with_capacity(total_rows);
        for chunk in 0..chunks {let start=chunk*local+job*local_rows;matrix.extend_from_slice(&values[start..start+local_rows]);}
        output.push(matrix);
    }Ok(output)
}
#[cfg(any(target_os="linux",test))]
fn gathered_mixed(values:&[f32],rows:&[Rows])->Result<Vec<Vec<f32>>> {
    if rows.is_empty() || rows.iter().any(|row|row.count==0 || row.total==0 || row.total%row.count!=0 || !row.split && row.total!=row.count) {return Err("resident mixed batch geometry differs".into());}
    let chunks=rows.iter().find(|row|row.split).map_or(1,|row|row.total/row.count);
    if rows.iter().any(|row|row.split && row.total/row.count!=chunks) {return Err("resident mixed batch row groups differ".into());}
    let stride=rows.iter().try_fold(0usize,|total,row|total.checked_add(row.count)).ok_or("resident mixed batch extent overflow")?;
    if stride.checked_mul(chunks)!=Some(values.len()) {return Err("resident mixed batch rank-major extent differs".into());}
    let mut offset=0;let mut output=Vec::new();
    for row in rows {let mut matrix=Vec::with_capacity(row.total);
        for rank in 0..if row.split {chunks} else {1} {let start=rank*stride+offset;matrix.extend_from_slice(&values[start..start+row.count]);}
        output.push(matrix);offset+=row.count;
    }Ok(output)
}

#[cfg(any(target_os="linux",test))]
#[path="resident_weights/batch_layout.rs"] mod batch_layout;

#[cfg(any(target_os="linux",test))]
#[path="resident_weights/load_rows.rs"] mod load_rows;

/// Bytes and matrices of the resident BF16 copy kept beside the FP32 copy (MEASURING FORM: on in code, no switch; the shipping ticket moves every reader to BF16 and drops the FP32 copies); printed in TEXT ADMISSION.
pub static BF16_COPY_BYTES:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
pub static BF16_COPY_MATRICES:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);

#[cfg(target_os="linux")]
mod device {
    use super::*;
    use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan}},safetensors::Dtype};
    use spark_gpu_probe::{driver::{Backend,DeviceVector},norm_boundary::WeightType,expert_ops::{Shape,Op}};
    use crate::{mxfp8_projection::Mxfp8Projection,dynamic_projection::ResidentDynamicProjection,transport::Ring};
    use std::collections::{BTreeMap,BTreeSet};
    mod route_device;
    mod projection_stage;
    pub use projection_stage::gather_stage;
    enum Storage<'b,'d> {Mx(ResidentDynamicProjection<'b,'d>),Plain {shape:spark_gpu_probe::causal_ops::Shape,weight:DeviceVector<'d>,bf16:Option<DeviceVector<'d>>},
        Packed {shape:Shape,weight:DeviceVector<'d>,scales:DeviceVector<'d>}}
    struct Matrix<'b,'d> {rows:Rows,cols:usize,storage:Storage<'b,'d>}
    struct Prepared {rows:Rows,cols:usize,upload:Option<engine_model::plan::admission::UploadWeight>,timings:engine_model::plan::admission::UploadTimings}
    #[derive(Default)]
    struct LoadTimings {payload:engine_model::plan::admission::UploadTimings,prepare_ns:u128,convert_ns:u128,upload_ns:u128,mxfp8_ns:u128}
    pub struct Weights<'b,'d> {matrices:BTreeMap<String,Matrix<'b,'d>>,embedding:Option<spark_gpu_probe::driver::Matrix<'d>>,
        ring:Option<Ring>,mailbox:Option<&'b spark_gpu_probe::driver::TpMailbox<'d>>,mhc_host:Option<spark_gpu_probe::driver::MhcHostLink<'b,'d>>,sequence:u64,topology:Topology,identity:Option<usize>,cost:crate::resident_profile::Cost,timings:[[u128;3];5],lut:Option<DeviceVector<'d>>,exchange:Option<DeviceVector<'d>>,small:std::cell::Cell<bool>,routes:BTreeMap<String,Option<route_device::Group<'d>>>,collecting:std::rc::Rc<std::cell::Cell<bool>>}
    fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {
        let payload=load_rows::padded_bytes(bytes);
        let mut out=b.vector(payload.len()/4)?;b.write_vector_bytes(&mut out,&payload)?;b.synchronize()?;out.into_resident_shared()
    }
    /// Local rows and each peer's rows go straight to their final job ranges. The
    /// owned queue retains the dropped local and output allocations. Both the copy
    /// and the mailbox operations keep their original ordered-stream order.
    fn gather_device<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&DeviceVector<'d>,rows:&[Rows])->Result<Vec<DeviceVector<'d>>> {
        let layout=super::batch_layout::Layout::new(rows)?;
        // Rows are immutable admission metadata from Rows::new on this owner.
        // The layout copies equal contiguous chunks; it selects no box-count path.
        topology.validate().map_err(|error|error.to_string())?;
        if local.len()!=layout.stride {return Err("resident device batch local extent differs".into());}
        let next=sequence.checked_add(1).filter(|&n|n<=u32::MAX as u64).ok_or("resident mailbox sequence exhausted")?;
        if mailbox.failed() {b.synchronize()?;return Err("resident device exchange failed".into());}
        layout.stride.checked_mul(layout.chunks).ok_or("resident device batch extent overflow")?;
        mailbox.layout().world.eq(&(topology.parts as usize)).then_some(()).ok_or("resident device batch mailbox topology differs")?;
        let (_,remote)=layout.placement(topology.rank as usize,topology.parts as usize)?;
        for (row,ranges) in rows.iter().zip(&layout.ranges) {let words=ranges.iter().map(|range|range.len()).sum::<usize>();
            if words!=row.total || words==0 {return Err("resident device batch output extent differs".into());}}
        let mut joined=b.vector(layout.output_words()?)?;
        // A failed take leaves its ranges unchanged while queued consumers may
        // execute before the sticky error is checked. Keep them initialized.
        b.zero_vector(&mut joined)?;
        let local_rows=layout.scatter(topology.rank as usize)?;
        if !local_rows.is_empty() {b.resident_scatter_copy(local,&mut joined,&local_rows)?;}
        // Publish the identical contiguous local payload, without rank-major staging.
        b.tp_mailbox_put_finite(mailbox,local,1,layout.stride,layout.stride,0,next)?;
        for (origin,_) in remote {let channel=mailbox.layout().channel(origin,topology.rank as usize);
            let placements=layout.scatter(origin)?;
            if placements.is_empty() {return Err("resident scatter origin has no split rows".into());}
            b.tp_mailbox_take_scatter(mailbox,&mut joined,&placements,channel,next,10_000_000_000)?;
        }*sequence=next;
        let joined=joined.into_resident_shared()?;let extents=rows.iter().map(|row|row.total).collect::<Vec<_>>();
        let mut output=Vec::with_capacity(rows.len());let mut start=0;
        for words in extents {output.push(joined.resident_range(start..start+words)?);start+=words;}Ok(output)
    }
    pub fn gather_argmax<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&DeviceVector<'d>,owners:&[Rows])->Result<u32> {
        let parts=topology.parts as usize;let total=parts.checked_mul(3).ok_or("resident argmax exchange overflow")?;
        if local.len()!=3 || owners.len()!=parts {return Err("resident argmax part extent differs".into());}
        let rows=Rows {start:topology.rank as usize*3,count:3,total,split:true};
        let gathered=gather_device(b,mailbox,sequence,topology,local,&[rows])?;
        b.read_argmax_raw(&gathered[0],parts,|values|merge_argmax(values,owners))
    }
    impl<'b,'d> Weights<'b,'d> {
        pub fn new(ring:Option<Ring>,cost:bool)->Self {
            let topology=ring.as_ref().map_or(Topology {rank:0,parts:1},|r|Topology {rank:r.rank() as u32,parts:r.world() as u32});
            Self {matrices:BTreeMap::new(),embedding:None,ring,mailbox:None,mhc_host:None,sequence:0,topology,identity:None,cost:crate::resident_profile::Cost::new(cost),timings:[[0;3];5],lut:None,exchange:None,small:std::cell::Cell::new(false),routes:BTreeMap::new(),collecting:std::rc::Rc::new(std::cell::Cell::new(false))}
        }
        pub fn attach_mailbox(&mut self,mailbox:&'b spark_gpu_probe::driver::TpMailbox<'d>)->Result<()> {
            let physical=Topology {rank:self.topology.rank,parts:mailbox.layout().world.try_into().map_err(|_|"mailbox topology exceeds ABI")?};
            let compatible=self.topology.eq(&physical);
            if !compatible || self.identity.is_some() || self.mailbox.is_some() {
                return Err("resident device exchange requires a fresh pair owner".into());
            }
            mailbox.claim_copy_group()?;self.mailbox=Some(mailbox);Ok(())
        }
        pub fn attach_mhc_host(&mut self,link:spark_gpu_probe::driver::MhcHostLink<'b,'d>)->Result<()> {
            if self.identity.is_some() || self.mhc_host.is_some() {return Err("host mHC link requires an unprepared resident owner".into());}
            link.require_success()?;self.mhc_host=Some(link);Ok(())
        }
        #[cfg(feature="kernel-bench")]
        pub fn fixture_vote(&mut self,ok:bool)->Result<bool> {
            let ring=self.ring.as_mut().ok_or("warm fixture requires its pair ring")?;
            let mut votes=vec![0u8;ring.world()];votes[ring.rank()]=u8::from(ok);
            ring.all_gather(&mut votes,1)?;Ok(votes.iter().all(|&vote|vote==1))
        }
        pub fn prepared(&self)->bool {self.identity.is_some()}
        pub fn prepare(&mut self,c:&CheckpointAdmission,b:&'b Backend<'d>)->Result<()> {
            let identity=std::ptr::from_ref(c) as usize;
            if let Some(prior)=self.identity {return if prior==identity {Ok(())} else {Err("resident checkpoint owner changed".into())};}
            let mut names=BTreeSet::new();names.insert("lm_head.weight".to_owned());
            for layer in 0..c.description().layers.len() {
                let plan=LayerPlan::new(c.description(),layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
                for step in plan.steps {if matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown|Kind::ExpandKv) {
                    for weight in step.weights {names.insert(weight.name);}
                }}
            }
            let mut bytes=0usize;
            for name in &names {let w=c.headers().weights().get(name).ok_or("resident matrix is absent")?;let logical=w.logical();let rows=Rows::new(&logical.shape,logical.storage,self.topology)?;
                let cols=usize::try_from(logical.shape[1]).map_err(|_|"resident columns exceed address space")?;
                let n=match logical.storage {
                    WeightStorage::Mxfp8=>rows.count.checked_mul(cols+cols/32),
                    WeightStorage::Nvfp4 {..}=>rows.count.checked_mul(cols/2+cols/16),
                    WeightStorage::Fp8Block {block}=>rows.count.checked_mul(cols).and_then(|n|n.checked_add(rows.count.div_ceil(block[0] as usize)*cols.div_ceil(block[1] as usize)*4)),
                    WeightStorage::Plain=>rows.count.checked_mul(cols).and_then(|n|n.checked_mul(if w.components().first().map(|c|c.dtype)==Some(Dtype::Bf16) && bf16_copy_wanted(name,rows.count,cols) {6} else {4})),
                }.ok_or("resident byte count overflow")?;bytes=bytes.checked_add(n).ok_or("resident total overflow")?;
            }
            let embedding_name="model.language_model.embed_tokens.weight";
            let embedding_view=c.headers().weights().get(embedding_name).ok_or("missing resident embedding")?;
            let d=c.description();
            if d.vocab>1<<24 || embedding_view.logical().shape!=[d.vocab,d.hidden] || embedding_view.logical().storage!=WeightStorage::Plain
                || embedding_view.components().len()!=1 || embedding_view.components()[0].dtype!=Dtype::Bf16 {return Err("resident embedding layout exceeds its checked ABI".into());}
            let embedding_bytes=d.vocab.checked_mul(d.hidden).and_then(|n|n.checked_mul(2)).and_then(|n|usize::try_from(n).ok()).ok_or("resident embedding size overflow")?;
            bytes=bytes.checked_add(embedding_bytes).ok_or("resident total overflow")?;
            let available=b.available_memory_bytes()?;
            if bytes.checked_add(1<<30).is_none_or(|need|need>available) {return Err(format!("resident matrices need {bytes} bytes plus 1073741824 bytes scratch; available={available}; parts={}",self.topology.parts));}
            eprintln!("RESIDENT LOAD rank={} parts={} matrices={} planned_bytes={bytes} available_bytes={available}",self.topology.rank,self.topology.parts,names.len());
            let started=std::time::Instant::now();let mut load=LoadTimings::default();let mut barrier_ns=0;
            let names=names.into_iter().collect::<Vec<_>>();
            let workers=std::thread::available_parallelism().map_or(1,usize::from).min(8);
            let sizes=names.iter().map(|name|c.headers().weights()[name].components().iter().try_fold(0u64,|bytes,component|bytes.checked_add(component.bytes).ok_or("resident host payload extent overflow"))).collect::<std::result::Result<Vec<_>,_>>()?;
            let mut begin=0;
            while begin<names.len() {
                let end=load_rows::batch_end(&sizes,begin,workers,256<<20);let batch=&names[begin..end];
                let topology=self.topology;
                let prepare=std::time::Instant::now();
                let prepared=load_rows::ordered(batch,|name|Matrix::read(c,name,topology).map_err(|error|format!("{name}: {error}")))?;
                load.prepare_ns+=prepare.elapsed().as_nanos();
                for (offset,(name,payload)) in batch.iter().zip(prepared).enumerate() {
                let index=begin+offset;
                let matrix=Matrix::load(c,name,self.topology,b,payload,&mut load).map_err(|e|format!("{name}: {e}"))?;self.matrices.insert(name.clone(),matrix);
                if index%256==0 {eprintln!("RESIDENT LOAD rank={} matrices={} allocated_bytes={}",self.topology.rank,index+1,b.allocated_bytes());
                    // Loading can exceed the normal exchange deadline. Keep peers
                    // together at bounded batches before any token transaction.
                    if let Some(ring)=self.ring.as_mut() {let wait=std::time::Instant::now();ring.barrier()?;barrier_ns+=wait.elapsed().as_nanos();}
                }
                }
                begin=end;
            }
            let embedding_started=std::time::Instant::now();let file=c.component_file(embedding_name).map_err(|e|e.to_string())?;let tensor=&file.header().tensors[embedding_name];
            let offset=file.header().data_start.checked_add(tensor.start).ok_or("resident embedding file offset overflow")?;
            let ranges=[spark_gpu_probe::load::WeightRange {file_offset:offset,destination:0,bytes:embedding_bytes,row_bytes:0,row_stride:0}];
            let arena=b.load_weight_arena(embedding_bytes,&[spark_gpu_probe::load::WeightFile {file:file.file(),ranges:&ranges}])?;
            self.embedding=Some(b.weight_matrix(&arena,0,d.vocab as usize,d.hidden as usize,spark_gpu_probe::load::WeightDtype::Bf16)?);
            b.synchronize()?;let embedding_ns=embedding_started.elapsed().as_nanos();if let Some(r)=self.ring.as_mut() {let wait=std::time::Instant::now();r.barrier()?;barrier_ns+=wait.elapsed().as_nanos();}
            self.identity=Some(identity);
            let total_ns=started.elapsed().as_nanos();let p=&load.payload;
            let measured=load.prepare_ns+load.convert_ns+load.upload_ns+load.mxfp8_ns+embedding_ns+barrier_ns;
            for (phase,ns) in [("payload_prepare_wall",load.prepare_ns),("plain_convert",load.convert_ns),("device_upload",load.upload_ns),("mxfp8_read_pack_upload",load.mxfp8_ns),("embedding_read_upload",embedding_ns),("barrier",barrier_ns),("other",total_ns.saturating_sub(measured))] {
                eprintln!("RESIDENT LOAD PHASE rank={} name={phase} nanoseconds={ns}",self.topology.rank);
            }
            for (phase,ns) in [("host_allocate_zero",p.allocate_ns),("payload_read",p.read_ns),("csf_unpack",p.unpack_ns),("payload_validation",p.validation_ns)] {
                eprintln!("RESIDENT LOAD WORK rank={} name={phase} nanoseconds={ns} aggregation=worker_sum_not_exclusive_wall",self.topology.rank);
            }
            eprintln!("RESIDENT LOAD PAYLOAD rank={} bytes={} content_hashing_ns=0 total_ns={total_ns} workers={workers}",self.topology.rank,p.bytes);
            eprintln!("RESIDENT READY rank={} parts={} matrices={} matrix_bytes={bytes} allocated_bytes={} bf16_copy_bytes={} bf16_copy_matrices={} load_seconds={:.6}",self.topology.rank,self.topology.parts,self.matrices.len(),b.allocated_bytes(),BF16_COPY_BYTES.load(std::sync::atomic::Ordering::Relaxed),BF16_COPY_MATRICES.load(std::sync::atomic::Ordering::Relaxed),started.elapsed().as_secs_f64());Ok(())
        }
        /// More matrices of the same checkpoint beside the loaded ones: the appended predictor
        /// layer, only when a caller asks for it. Device bytes are counted as every allocation is.
        pub fn prepare_names(&mut self,c:&CheckpointAdmission,b:&'b Backend<'d>,names:&[String])->Result<()> {
            if self.identity!=Some(std::ptr::from_ref(c) as usize) {return Err("resident owner is not prepared for this checkpoint".into());}
            let started=std::time::Instant::now();let mut load=LoadTimings::default();let before=b.allocated_bytes();let mut loaded=0usize;let mut borrowed=0usize;
            for name in names {if self.matrices.contains_key(name) {continue;}
                let mut payload=Matrix::read(c,name,self.topology).map_err(|error|format!("{name}: {error}"))?;
                // The appended layer's experts are stored without an activation scale: their source
                // takes unquantized activations, and our NVFP4 kernels quantize them. The scale of
                // the same expert tensor in the last decoder layer stands in. This is an
                // approximation of the proposer only: it can lower the kept rate, never change a token.
                if let Some(upload)=payload.upload.as_mut().filter(|u|matches!(u.storage,WeightStorage::Nvfp4 {..}) && u.input_scale.is_none()) {
                    let layers=c.description().layers.len();let donor=name.replacen(&format!(".layers.{layers}."),&format!(".layers.{}.",layers-1),1);
                    if donor==*name {return Err(format!("{name}: resident calibrated scales are absent"));}
                    let (other,_)=c.upload_rows_timed(&donor,0..1).map_err(|error|format!("{donor}: {error}"))?;
                    upload.input_scale=Some(other.input_scale.ok_or_else(||format!("{donor}: activation scale is absent"))?);borrowed+=1;
                }
                let matrix=Matrix::load(c,name,self.topology,b,payload,&mut load).map_err(|e|format!("{name}: {e}"))?;self.matrices.insert(name.clone(),matrix);loaded+=1;
                if loaded.is_multiple_of(128)&& let Some(ring)=self.ring.as_mut() {ring.barrier()?;}
            }
            if loaded>0 {b.synchronize()?;if let Some(ring)=self.ring.as_mut() {ring.barrier()?;}
                eprintln!("RESIDENT PREDICTOR rank={} matrices={loaded} borrowed_activation_scales={borrowed} device_bytes={} allocated_bytes={} load_seconds={:.6}",self.topology.rank,b.allocated_bytes().saturating_sub(before),b.allocated_bytes(),started.elapsed().as_secs_f64());}
            Ok(())
        }
        pub fn load_for_proof(c:&CheckpointAdmission,b:&'b Backend<'d>,topology:Topology)->Result<Self> {
            topology.validate().map_err(|error|error.to_string())?;let mut weights=Self::new(None,false);
            weights.topology=topology;weights.prepare(c,b)?;Ok(weights)
        }
        #[cfg(feature="kernel-bench")]
        pub fn load_local_fixture(c:&CheckpointAdmission,b:&'b Backend<'d>,topology:Topology,names:&[String])->Result<Self> {
            topology.validate().map_err(|error|error.to_string())?;
            let mut weights=Self::new(None,false);weights.topology=topology;
            weights.identity=Some(std::ptr::from_ref(c) as usize);weights.collecting.set(true);
            weights.prepare_names(c,b,names)?;Ok(weights)
        }
        pub fn visit_device_bytes(self,b:&Backend<'d>,mut visit:impl FnMut(&str,&str,usize,usize,Vec<u8>)->Result<()>)->Result<()> {
            let mut storage=|name:&str,part:&str,total:usize,read:&mut dyn FnMut(usize,usize)->Result<Vec<u8>>|->Result<()> {
                if !total.is_multiple_of(4) {return Err("resident proof storage is not word aligned".into());}
                for offset in (0..total).step_by(8<<20) {let count=(8<<20).min(total-offset);visit(name,part,offset,total,read(offset,count)?)?;}Ok(())
            };
            let read_vector=|vector:&DeviceVector<'d>,offset,count|->Result<Vec<u8>> {
                Ok(b.read_vector_values(&vector.resident_range(offset/4..(offset+count)/4)?)?.into_iter().flat_map(|value|value.to_bits().to_le_bytes()).collect())
            };
            for (name,matrix) in self.matrices {match matrix.storage {
                Storage::Plain {weight,..}=>{let weight=weight.into_resident_shared()?;storage(&name,"weight",weight.len()*4,&mut |offset,count|read_vector(&weight,offset,count))?;},
                Storage::Packed {weight,scales,..}=>{
                    let weight=weight.into_resident_shared()?;let scales=scales.into_resident_shared()?;
                    storage(&name,"weight",weight.len()*4,&mut |offset,count|read_vector(&weight,offset,count))?;
                    storage(&name,"scales",scales.len()*4,&mut |offset,count|read_vector(&scales,offset,count))?;
                },
                Storage::Mx(matrix)=>{let resident=matrix.resident();let sizes=resident.storage_bytes();
                    storage(&name,"weight",sizes[0],&mut |offset,count|resident.read_storage(false,offset,count))?;
                    storage(&name,"scales",sizes[1],&mut |offset,count|resident.read_storage(true,offset,count))?;
                },
            }}
            let embedding=self.embedding.as_ref().ok_or("resident proof embedding is absent")?;
            storage("model.language_model.embed_tokens.weight","weight",b.matrix_storage_bytes(embedding),&mut |offset,count|b.read_matrix_storage(embedding,offset,count))
        }
        pub fn embedding(&self,b:&Backend<'d>,token:u32,streams:usize)->Result<DeviceVector<'d>> {
            let table=self.embedding.as_ref().ok_or("embedding is not resident")?;
            if token as usize>=table.rows() || streams==0 {return Err("resident embedding ID or stream count differs".into());}
            // One token starts here. Leave 1 GiB under the vector pool's cap for its buffers
            // (54 percent of the least cap of 2,000,000,000 bytes; less of a raised cap).
            b.trim_vector_spares(1<<30);
            // A decode token keeps its own gathers as they are. Only small row groups change.
            self.small.set(false);
            let ids=b.upload_vector(&vec![token as f32;streams])?;let mut out=b.vector(streams.checked_mul(table.cols()).ok_or("resident embedding output overflow")?)?;
            // An owned queue keeps the dropped ids alive until its real drain, so a
            // decode token does not start with a wait. Ordinary callers still wait.
            let launched=b.batch_embedding_device(table,&ids,&mut out);let completed=if b.resident_queue_active() {Ok(())} else {b.synchronize()};launched.and(completed)?;Ok(out)
        }
        pub fn embedding_rows(&self,b:&Backend<'d>,tokens:&[u32],streams:usize)->Result<DeviceVector<'d>> {
            self.small.set(tokens.len()<crate::bounded_decoder::PROMPT_SHARED);
            let table=self.embedding.as_ref().ok_or("embedding is not resident")?;
            if tokens.iter().any(|&t|t as usize>=table.rows()) || streams==0 {return Err("resident embedding ID or stream count differs".into());}
            let ids=tokens.iter().flat_map(|&t|std::iter::repeat_n(t as f32,streams)).collect::<Vec<_>>();
            let ids=b.upload_vector(&ids)?;let mut out=b.vector(ids.len().checked_mul(table.cols()).ok_or("resident embedding output overflow")?)?;
            let launched=b.batch_embedding_device(table,&ids,&mut out);let completed=b.synchronize();launched.and(completed)?;Ok(out)
        }
        /// Every rank must make the same eviction and collective-skip decisions.
        pub fn shared_budget(&mut self,local:usize)->Result<usize> {
            let Some(ring)=self.ring.as_mut() else {return Ok(local);};
            let mut bytes=vec![0;ring.world()*8];let at=ring.rank()*8;bytes[at..at+8].copy_from_slice(&(local as u64).to_le_bytes());ring.all_gather(&mut bytes,8)?;
            let minimum=bytes.as_chunks::<8>().0.iter().map(|v|u64::from_le_bytes(*v)).min().ok_or("empty resident budget agreement")?;
            usize::try_from(minimum).map_err(|_|"resident shared budget exceeds address space".into())
        }
        /// CHAIN-SHADOW: `bytes` (at most to the end) from `offset` of a resident matrix's weight
        /// storage, or None when `name` is not resident yet or `offset` is at or past its end.
        /// Scales are not included. Also returns the matrix's weight bytes.
        pub fn shadow_range<'v>(&'v self,b:&Backend<'d>,name:&str,offset:usize,bytes:usize)->Result<Option<(spark_gpu_probe::driver::ShadowRange<'v>,usize)>> {
            let Some(matrix)=self.matrices.get(name) else {return Ok(None)};
            let total=match &matrix.storage {Storage::Mx(projection)=>projection.resident().weight_bytes(),Storage::Plain {weight,..}|Storage::Packed {weight,..}=>weight.len()*4};
            if offset>=total || bytes==0 {return Ok(None);}let bytes=bytes.min(total-offset);
            Ok(Some((match &matrix.storage {
                Storage::Mx(projection)=>projection.resident().shadow_range(offset,bytes)?,
                Storage::Plain {weight,..}|Storage::Packed {weight,..}=>b.shadow_vector_range(weight,offset,bytes)?,
            },total)))
        }
        fn project_local(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,round:bool)->Result<(DeviceVector<'d>,Rows,usize)> {
            let stamp=self.cost.start(b);let started=std::time::Instant::now();
            let matrix=self.matrices.get(name).ok_or_else(||format!("projection is not resident: {name}"))?;
            if input.len()!=matrix.cols || input.is_segmented() {return Err("resident projection feature extent differs".into());}
            let class=match &matrix.storage {Storage::Mx(_)=>0,Storage::Plain {..}=>1,Storage::Packed {shape,..}=>if shape.op==Op::NvProject {2} else {3}};
            let output=match &matrix.storage {
                Storage::Mx(p)=>p.project_vector(input)?.into_output().into_vector(),
                Storage::Plain {shape,weight,bf16}=>match bf16 {Some(packed) if shape.op==spark_gpu_probe::causal_ops::Op::Project => b.project_bf16(shape.rows as usize,shape.cols as usize,input,packed)?,_=>b.causal_operation(*shape,[Some(input),None,None,None],Some(weight))?},
                Storage::Packed {shape,weight,scales}=>{
                    let packed=if shape.op==Op::NvProject {Some(b.expert_operation(Shape {op:Op::Quantize,..*shape},[Some(input),None,None,None],None,None)?)} else {None};
                    b.expert_operation(*shape,[Some(packed.as_ref().unwrap_or(input)),None,None,None],Some(weight),Some(scales))?
                },
            };
            // Packed projection kernels and the typed MX output already store BF16
            // lanes. A second cast is an identity for every stored BF16 pattern.
            // Plain ordered projections still require their explicit boundary.
            let rounded=matches!(&matrix.storage,Storage::Mx(_)|Storage::Packed {shape:Shape {op:Op::NvProject|Op::Fp8Project|Op::Fp8Serial,..},..});
            let output=if round && !rounded {b.round_bf16_lanes(&output)?.into_vector()} else {output};
            let bytes=match &matrix.storage {Storage::Mx(_)=>matrix.rows.count*(matrix.cols+matrix.cols/32),Storage::Plain {weight,bf16,..}=>bf16.as_ref().map_or(weight.len()*4,|packed|packed.len()*4),Storage::Packed {weight,scales,..}=>(weight.len()+scales.len())*4};
            self.cost.record(||if name=="lm_head.weight" {"projection_head".into()} else {format!("projection_{class}")},stamp,b,bytes);
            self.timings[class][0]+=1;self.timings[class][1]+=started.elapsed().as_nanos();Ok((output,matrix.rows,class))
        }
        pub fn apply(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,round:bool)->Result<DeviceVector<'d>> {
            let (output,rows,class)=self.project_local(name,b,input,round)?;
            if !rows.split || self.collecting.get() {return Ok(output);}let started=std::time::Instant::now();
            if let Some(mailbox)=self.mailbox {
                let stamp=self.cost.start(b);
                // The shared gather copies these same local words to their final
                // range and sends them directly. Its fresh fill and scatter are
                // one launch; failed takes still leave initialized peer ranges.
                let mut values=gather_device(b,mailbox,&mut self.sequence,self.topology,&output,&[rows])?;
                let gathered=values.pop().ok_or("resident projection gather has no output")?;
                self.cost.record(||"exchange_device".into(),stamp,b,0);
                self.timings[class][2]+=started.elapsed().as_nanos();
                return Ok(gathered);
            }
            let stamp=self.cost.start(b);let local=b.read_vector_values(&output)?;
let mut bytes=vec![0;rows.total*4];
            for (dst,v) in bytes[rows.start*4..(rows.start+rows.count)*4].as_chunks_mut::<4>().0.iter_mut().zip(local) {dst.copy_from_slice(&v.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);let stamp=self.cost.start(b);
            self.ring.as_mut().ok_or("resident row split has no group")?.all_gather(&mut bytes,4)?;
            self.cost.record(||"exchange_network".into(),stamp,b,0);let stamp=self.cost.start(b);
            let values=bytes.as_chunks::<4>().0.iter().map(|v|f32::from_le_bytes(*v)).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) {return Err("resident gathered projection is not finite".into());}
            let output=b.upload_vector(&values)?;b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);self.timings[class][2]+=started.elapsed().as_nanos();Ok(output)
        }
        pub fn apply_greedy(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>)->Result<u32> {
            let rows=self.matrices.get(name).ok_or_else(||format!("projection is not resident: {name}"))?.rows;
            if rows.count>1<<24 {let output=self.apply(name,b,input,true)?;return crate::bounded_decoder::greedy_logits(&b.read_vector_values(&output)?,rows.total);}
            let (output,rows,_)=self.project_local(name,b,input,true)?;
            let mut partials=b.vector(output.len().div_ceil(1024)*3)?;let mut result=b.vector(3)?;
            b.enqueue_argmax_device(&output,&mut partials,&mut result,0)?;
            if !rows.split {return b.read_argmax_raw(&result,1,|values|merge_argmax(values,&[rows]));}
            let parts=self.topology.parts as usize;
            if rows.count.checked_mul(parts)!=Some(rows.total) || rows.start!=rows.count*self.topology.rank as usize {return Err("resident argmax ownership differs".into());}
            let owners=(0..parts).map(|rank|Rows {start:rank*rows.count,..rows}).collect::<Vec<_>>();
            if let Some(mailbox)=self.mailbox {let stamp=self.cost.start(b);
                let selected=gather_argmax(b,mailbox,&mut self.sequence,self.topology,&result,&owners);
                self.cost.record(||"exchange_argmax".into(),stamp,b,0);return selected;
            }
            self.gather_mapped(b,&result)?;let gathered=self.exchange.as_ref().ok_or("resident argmax exchange is absent")?;
            b.read_argmax_raw(gathered,parts,|values|merge_argmax(values,&owners))
        }
        #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
        pub fn apply_batch(&mut self,b:&Backend<'d>,input:&[(&str,&DeviceVector<'_>)],round:bool)->Result<Option<Vec<DeviceVector<'d>>>> {
            if input.is_empty() {return Err("resident projection batch is empty".into());}
            let matrices=input.iter().map(|(name,_)|self.matrices.get(*name).ok_or_else(||format!("projection is not resident: {name}"))).collect::<Result<Vec<_>>>()?;
            if input.len()<=9 && matrices.iter().all(|matrix|matches!(&matrix.storage,Storage::Mx(_)|Storage::Plain {..}) && matrix.cols%32==0) {
                return self.apply_mixed_batch(b,input,round);
            }
            let first=matrices[0];
            // One op for the whole batch: NVFP4 (quantized activations) or block FP8 (the kernel reads the FP32 activation as it is).
            let op=match &first.storage {Storage::Packed {shape,..} if matches!(shape.op,Op::NvProject|Op::Fp8Serial)=>shape.op,_=>return Ok(None)};
            // Block FP8 batches only inside a projection stage (`collecting`): on one Spark the old single launches stay, with no host read, upload or wait.
            if op==Op::Fp8Serial && !self.collecting.get() {return Ok(None);}
            if matrices.iter().any(|matrix|matrix.rows!=first.rows || matrix.cols!=first.cols || !matches!(&matrix.storage,Storage::Packed {shape,..} if shape.op==op)) {return Ok(None);}
            if input.iter().any(|(_,value)|value.len()!=first.cols || value.is_segmented()) {return Err("resident batch feature extent differs".into());}
            let stamp=self.cost.start(b);let started=std::time::Instant::now();
            let mut quantized=Vec::new();let mut indices=Vec::new();
            let weight_bytes=matrices.iter().map(|matrix|{let Storage::Packed {weight,scales,..}=&matrix.storage else {unreachable!()};(weight.len()+scales.len())*4}).sum::<usize>();
            // Block FP8 has no Quantize step: every job reads its FP32 input as it is, exactly as `apply` passes it.
            if op==Op::NvProject {
                // Distinct inputs are quantized together: one launch, one output vector, one view a job (the same bytes as one launch each).
                let mut pending:Vec<((u64,usize,u32),Shape,&DeviceVector<'_>)>=Vec::new();
                for (matrix,(_,value)) in matrices.iter().zip(input) {
                    let Storage::Packed {shape,..}=&matrix.storage else {unreachable!()};
                    let key=(value.device_address(),matrix.cols,shape.activation.to_bits());
                    let index=if let Some(index)=pending.iter().position(|(prior,_,_)|*prior==key) {index} else {pending.push((key,Shape {op:Op::Quantize,..*shape},*value));pending.len()-1};
                    indices.push(index);
                }
                if spark_gpu_probe::quantize_sweep::use_batch(&pending.iter().map(|(_,shape,_)|shape.cols).collect::<Vec<_>>()) && pending.iter().all(|(_,shape,_)|spark_gpu_probe::expert_ops::quantize_lanes_admitted(shape)) {
                    let jobs=pending.iter().map(|(_,shape,value)|(*shape,*value)).collect::<Vec<_>>();
                    let (all,each)=b.expert_quantize_batch(&jobs)?;let all=all.into_resident_shared()?;
                    for index in 0..pending.len() {quantized.push((pending[index].0,all.resident_range(index*each..(index+1)*each)?));}
                } else {
                    for (key,shape,value) in &pending {quantized.push((*key,b.expert_operation(*shape,[Some(value),None,None,None],None,None)?));}
                }
            }
            let jobs=matrices.iter().zip(input).enumerate().map(|(job,(matrix,(_,value)))| {let Storage::Packed {shape,weight,scales}=&matrix.storage else {unreachable!()};
                (*shape,if op==Op::NvProject {&quantized[indices[job]].1} else {*value},weight,scales)
            }).collect::<Vec<_>>();
            let output=b.expert_projection_batch(&jobs)?;
            // The cast output is the value. No second copy of it is made. Block FP8 stores BF16 lanes already, so (as in `apply`) it needs no second cast.
            let output=if round && op==Op::NvProject {let rounded=b.round_bf16_lanes(&output)?.into_vector();if !b.resident_queue_active() {b.synchronize()?;}rounded} else {output};
            self.cost.record(||String::from(if op==Op::NvProject {"projection_2"} else {"projection_3"}),stamp,b,weight_bytes);let compute=started.elapsed().as_nanos();let stamp=self.cost.start(b);
            if self.collecting.get() {let output=output.into_resident_shared()?;
                return (0..input.len()).map(|job|output.resident_range(job*first.rows.count..(job+1)*first.rows.count)).collect::<Result<Vec<_>>>().map(Some);
            }
            if let Some(mailbox)=self.mailbox.filter(|_|first.rows.split) {
                let rows=vec![first.rows;input.len()];let result=gather_device(b,mailbox,&mut self.sequence,self.topology,&output,&rows)?;
                self.cost.record(||"exchange_device".into(),stamp,b,0);self.timings[2][0]+=input.len() as u128;self.timings[2][1]+=compute;self.timings[2][2]+=started.elapsed().as_nanos()-compute;return Ok(Some(result));
            }

            let local=b.read_vector_values(&output)?;let ranks=if first.rows.split {self.topology.parts as usize} else {1};
            let mut bytes=vec![0;local.len().checked_mul(ranks).and_then(|words|words.checked_mul(4)).ok_or("resident batch exchange overflow")?];
            let start=if first.rows.split {self.topology.rank as usize*local.len()*4} else {0};
            for (destination,value) in bytes[start..start+local.len()*4].as_chunks_mut::<4>().0.iter_mut().zip(local) {destination.copy_from_slice(&value.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);
            if first.rows.split {let stamp=self.cost.start(b);self.ring.as_mut().ok_or("resident batch has no row group")?.all_gather(&mut bytes,4)?;self.cost.record(||"exchange_network".into(),stamp,b,0);}
            let stamp=self.cost.start(b);let values=bytes.as_chunks::<4>().0.iter().map(|word|f32::from_le_bytes(*word)).collect::<Vec<_>>();
            if values.iter().any(|value|!value.is_finite()) {return Err("resident batch projection is not finite".into());}
            let result=gathered_batch(&values,first.rows.count,first.rows.total,input.len())?.iter().map(|values|b.upload_vector(values)).collect::<Result<Vec<_>>>()?;b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);self.timings[2][0]+=input.len() as u128;self.timings[2][1]+=compute;self.timings[2][2]+=started.elapsed().as_nanos()-compute;Ok(Some(result))
        }
        fn apply_mixed_batch(&mut self,b:&Backend<'d>,input:&[(&str,&DeviceVector<'_>)],round:bool)->Result<Option<Vec<DeviceVector<'d>>>> {
            use spark_gpu_probe::{driver::ResidentProjection,causal_ops::{Shape as CausalShape,Op as CausalOp}};
            let matrices=input.iter().map(|(name,_)|self.matrices.get(*name).ok_or_else(||format!("projection is not resident: {name}"))).collect::<Result<Vec<_>>>()?;
            if input.iter().zip(&matrices).any(|((_,value),matrix)|value.len()!=matrix.cols || value.is_segmented()) {return Err("resident mixed batch feature extent differs".into());}
            let stamp=self.cost.start(b);let started=std::time::Instant::now();let mut quantized=Vec::new();let mut indices=Vec::new();let mut weight_bytes=0usize;
            for (matrix,(_,value)) in matrices.iter().zip(input) {
                indices.push(if matches!(&matrix.storage,Storage::Mx(_)) {let key=(value.device_address(),matrix.cols);
                    Some(if let Some(index)=quantized.iter().position(|(prior,_)|*prior==key) {index} else {
                        let shape=CausalShape {op:CausalOp::Mxfp8QuantizePositiveZero,heads:0,dim:0,taps:0,rows:0,cols:matrix.cols.try_into().map_err(|_|"resident mixed batch columns exceed ABI")?,epsilon:0.0,bound:0.0};
                        quantized.push((key,b.causal_operation(shape,[Some(value),None,None,None],None)?));quantized.len()-1
                    })
                } else {None});
                weight_bytes+=match &matrix.storage {Storage::Mx(_)=>matrix.rows.count*matrix.cols+matrix.rows.count*matrix.cols/32,Storage::Plain {weight,..}=>weight.len()*4,_=>return Err("resident mixed batch storage differs".into())};
            }
            let jobs=matrices.iter().zip(input).zip(&indices).map(|((matrix,(_,value)),index)|match &matrix.storage {
                Storage::Mx(projection)=>ResidentProjection::Mxfp8 {matrix:projection.resident(),input:&quantized[index.unwrap()].1},
                Storage::Plain {weight,..}=>ResidentProjection::Plain {rows:matrix.rows.count,cols:matrix.cols,weight,input:value},_=>unreachable!(),
            }).collect::<Vec<_>>();
            let local_rows=matrices.iter().try_fold(0usize,|total,matrix|total.checked_add(matrix.rows.count)).ok_or("resident mixed batch rows overflow")?;
            let mut output=b.vector(local_rows)?;b.resident_projection_batch(&jobs,&mut output,round)?;
            self.cost.record(||"projection_mixed".into(),stamp,b,weight_bytes);let compute=started.elapsed().as_nanos();let stamp=self.cost.start(b);
            if self.collecting.get() {let output=output.into_resident_shared()?;let mut offset=0;
                return matrices.iter().map(|matrix| {let start=offset;offset+=matrix.rows.count;output.resident_range(start..offset)}).collect::<Result<Vec<_>>>().map(Some);
            }
            if let Some(mailbox)=self.mailbox.filter(|_|matrices.iter().any(|matrix|matrix.rows.split)) {
                let rows=matrices.iter().map(|matrix|matrix.rows).collect::<Vec<_>>();let result=gather_device(b,mailbox,&mut self.sequence,self.topology,&output,&rows)?;
                self.cost.record(||"exchange_device".into(),stamp,b,0);self.timings[4][0]+=input.len() as u128;self.timings[4][1]+=compute;self.timings[4][2]+=started.elapsed().as_nanos()-compute;return Ok(Some(result));
            }

            let local=b.read_vector_values(&output)?;let split=matrices.iter().any(|matrix|matrix.rows.split);let ranks=if split {self.topology.parts as usize} else {1};
            let mut bytes=vec![0;local.len().checked_mul(ranks).and_then(|words|words.checked_mul(4)).ok_or("resident mixed batch exchange overflow")?];
            let start=if split {self.topology.rank as usize*local.len()*4} else {0};
            for (destination,value) in bytes[start..start+local.len()*4].as_chunks_mut::<4>().0.iter_mut().zip(local) {destination.copy_from_slice(&value.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);
            if split {let stamp=self.cost.start(b);self.ring.as_mut().ok_or("resident mixed batch has no row group")?.all_gather(&mut bytes,4)?;self.cost.record(||"exchange_network".into(),stamp,b,0);}
            let stamp=self.cost.start(b);let values=bytes.as_chunks::<4>().0.iter().map(|word|f32::from_le_bytes(*word)).collect::<Vec<_>>();
            if values.iter().any(|value|!value.is_finite()) {return Err("resident mixed batch projection is not finite".into());}
            let rows=matrices.iter().map(|matrix|matrix.rows).collect::<Vec<_>>();
            let result=gathered_mixed(&values,&rows)?.iter().map(|values|b.upload_vector(values)).collect::<Result<Vec<_>>>()?;b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);self.timings[4][0]+=input.len() as u128;self.timings[4][1]+=compute;self.timings[4][2]+=started.elapsed().as_nanos()-compute;Ok(Some(result))
        }
        /// One immutable weight pass for independent prompt rows. The packed
        /// output is [weight row, token], so row ownership still stays contiguous.
        pub fn apply_rows(&mut self,name:&str,b:&Backend<'d>,input:&[&DeviceVector<'_>],round:bool)->Result<Vec<DeviceVector<'d>>> {
            if input.is_empty() || input.len()>crate::bounded_decoder::PROMPT_ROWS {return Err("resident prompt group exceeds its row bound".into());}
            let matrix=self.matrices.get(name).ok_or_else(||format!("projection is not resident: {name}"))?;
            if input.len()==1 || matches!(matrix.storage,Storage::Mx(_)) {return input.iter().map(|row|self.apply(name,b,row,round)).collect();}
            if input.iter().any(|row|row.len()!=matrix.cols || row.is_segmented()) {return Err("resident prompt feature extent differs".into());}
            let m=input.len();let stamp=self.cost.start(b);let mut joined=b.vector(m.checked_mul(matrix.cols).ok_or("prompt input overflow")?)?;
            for (row,value) in input.iter().enumerate() {b.copy_range(value,&mut joined,0,row*matrix.cols,matrix.cols)?;}b.synchronize()?;
            let (output,class)=match &matrix.storage {
                Storage::Plain {shape,weight,..}=>(b.causal_operation(spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::ProjectRows,heads:m as u32,..*shape},[Some(&joined),None,None,None],Some(weight))?,1),
                Storage::Packed {shape,weight,scales}=>{
                    let shape=Shape {op:if shape.op==Op::NvProject {Op::NvRows} else {Op::Fp8Rows},tokens:m as u32,..*shape};
                    let packed=if shape.op==Op::NvRows {Some(b.expert_operation(Shape {op:Op::QuantizeRows,..shape},[Some(&joined),None,None,None],None,None)?)} else {None};
                    (b.expert_operation(shape,[Some(packed.as_ref().unwrap_or(&joined)),None,None,None],Some(weight),Some(scales))?,if shape.op==Op::NvRows {2} else {3})
                },
                Storage::Mx(_)=>unreachable!(),
            };
            let output=if round {let rounded=b.round_bf16_lanes(&output)?;let mut copy=b.vector(output.len())?;b.copy_prefix(rounded.vector(),&mut copy,output.len())?;b.synchronize()?;copy} else {output};
            let bytes=match &matrix.storage {Storage::Plain {weight,..}=>weight.len()*4,Storage::Packed {weight,scales,..}=>(weight.len()+scales.len())*4,Storage::Mx(_)=>unreachable!()};
            self.cost.record(||format!("projection_{class}"),stamp,b,bytes);let stamp=self.cost.start(b);
            let local=b.read_vector_values(&output)?;let mut bytes=vec![0;matrix.rows.total*m*4];
            for (dst,value) in bytes[matrix.rows.start*m*4..(matrix.rows.start+matrix.rows.count)*m*4].as_chunks_mut::<4>().0.iter_mut().zip(local) {dst.copy_from_slice(&value.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);
            if matrix.rows.split {let stamp=self.cost.start(b);self.ring.as_mut().ok_or("resident row split has no group")?.all_gather(&mut bytes,m*4)?;self.cost.record(||"exchange_network".into(),stamp,b,0);}
            let stamp=self.cost.start(b);let values=bytes.as_chunks::<4>().0.iter().map(|v|f32::from_le_bytes(*v)).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) {return Err("resident prompt projection is not finite".into());}
            let mut result=Vec::new();for token in 0..m {let row=(0..matrix.rows.total).map(|column|values[column*m+token]).collect::<Vec<_>>();result.push(b.upload_vector(&row)?);}b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);Ok(result)
        }
        /// All ranks' equal local vectors, rank-major, in one mapped allocation. GB10 has
        /// one DRAM pool, so the wire writes it and later kernels read it, with no host
        /// copy. The allocation grows rarely and stays registered between transfers.
        /// The blocks of every rank in rank order, gathered on the device mailbox as a decode
        /// step gathers them: no wait for the stream and no host link step. For a decode token
        /// and for a small row group (a verify pass makes about 1,000 gathers; each host link
        /// step left the device idle 24 to 86 us, measured). None: the caller takes the host
        /// link, which large groups and blocks over one mailbox slot keep. Same words either way.
        fn gather_small(&mut self,b:&Backend<'d>,local:&DeviceVector<'d>)->Result<Option<DeviceVector<'d>>> {
            let parts=self.topology.parts as usize;let words=local.len();
            let Some(mailbox)=self.mailbox.filter(|m|self.small.get() && words>0 && words<=m.slot_bytes()/4) else {return Ok(None);};
            let stamp=self.cost.start(b);
            // One block for each rank, already in rank order: the packed vector is the result,
            // and the job-major placement that `gather_device` makes for several matrices is not needed.
            self.topology.validate().map_err(|error|error.to_string())?;
            mailbox.layout().world.eq(&parts).then_some(()).ok_or("prompt mailbox topology differs")?;
            let next=self.sequence.checked_add(1).filter(|&n|n<=u32::MAX as u64).ok_or("resident mailbox sequence exhausted")?;
            if mailbox.failed() {b.synchronize()?;return Err("resident device exchange failed".into());}
            let total=words.checked_mul(parts).ok_or("prompt exchange overflow")?;let rank=self.topology.rank as usize;let own=rank*words;
            // No clear: this rank's copy and one take for each other rank write every word, in
            // stream order, before anything reads the vector.
            let mut packed=b.vector(total)?;b.copy_range(local,&mut packed,0,own,words)?;
            b.tp_mailbox_put_finite(mailbox,&packed,1,total,words,own,next)?;
            for origin in (0..rank).chain(rank+1..parts) {let channel=mailbox.layout().channel(origin,rank);
                b.tp_mailbox_take_from(mailbox,&mut packed,1,total,words,origin*words,channel,next,10_000_000_000)?;
            }self.sequence=next;
            self.cost.record(||"exchange_device".into(),stamp,b,0);Ok(Some(packed))
        }
        fn gather_mapped(&mut self,b:&Backend<'d>,local:&DeviceVector<'_>)->Result<()> {
            let stamp=self.cost.start(b);
            let parts=self.topology.parts as usize;let words=local.len();let total=words.checked_mul(parts).ok_or("prompt exchange overflow")?;
            if self.exchange.as_mut().is_none_or(|v|v.resize_within_capacity(total).is_err()) {
                // The wire keeps a device allocation's registration until told to drop it.
                b.synchronize()?;if let Some(ring)=self.ring.as_mut() {ring.release_transfer_regions();}self.exchange=None;
                self.exchange=Some(b.mapped_vector(total.checked_mul(3).ok_or("prompt exchange overflow")?/2)?);
            }
            let exchange=self.exchange.as_mut().ok_or("prompt exchange is absent")?;
            b.copy_range(local,exchange,0,self.topology.rank as usize*words,words)?;
            // The link step's contract: every queued write to the blocks has finished.
            b.synchronize()?;
            self.cost.record(||"exchange_pack".into(),stamp,b,0);let stamp=self.cost.start(b);
            // The kernel that reads this buffer next refuses a value that is not finite, on the
            // device, with `first` set: a host scan of the mapped buffer took 41 percent of this row.
            crate::model::exchange_blocks(b,self.ring.as_mut().ok_or("resident row split has no group")?,exchange,parts,1,words,false)?;
            self.cost.record(||"exchange_network".into(),stamp,b,0);Ok(())
        }
        /// Token-major `[m,cols]` to token-major `[m,rows]`, complete on every rank.
        /// One weight pass serves all rows. Each element keeps its feature fold.
        pub fn apply_matrix(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,m:usize,round:bool)->Result<DeviceVector<'d>> {
            use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
            let stamp=self.cost.start(b);
            if self.lut.is_none() {self.lut=Some(b.upload_vector(&spark_gpu_probe::expert_ops::decode_lut())?);}
            let matrix=self.matrices.get(name).ok_or_else(||format!("projection is not resident: {name}"))?;
            let cols=matrix.cols;if m==0 || input.len()!=m.checked_mul(cols).ok_or("prompt input overflow")? || input.is_segmented() {return Err("resident prompt feature extent differs".into());}
            let local=matrix.rows.count;
            let transpose=|x:&DeviceVector<'_>,rows:usize,columns:usize|->Result<DeviceVector<'d>> {let mut s=RowShape::new(Row::Transpose,1).ok_or("prompt transpose shape")?;
                s.rows=u32::try_from(rows).map_err(|_|"prompt transpose exceeds ABI")?;s.cols=u32::try_from(columns).map_err(|_|"prompt transpose exceeds ABI")?;
                b.prompt_operation(s,[Some(x),None,None,None,None],None)};
            // Output in [local rows,m] when row-major, else [m,local rows].
            let (output,row_major,class)=match &matrix.storage {
                Storage::Mx(p)=>{let v=p.project_rows(input,m)?;let mut out=b.vector(m*local)?;b.copy_prefix(v.vector(),&mut out,m*local)?;(out,false,0)},
                Storage::Plain {shape,weight,..}=>{let mut s=RowShape::new(Row::ProjectRows,m).ok_or("prompt rows exceed ABI")?;s.rows=shape.rows;s.cols=shape.cols;
                    let out=b.prompt_operation(s,[Some(input),None,None,None,None],Some(weight))?;
                    // The cast output is the value. No second copy of it is made.
                    (if round {b.round_bf16_lanes(&out)?.into_vector()} else {out},true,1)},
                Storage::Packed {shape,weight,scales}=>{
                    let nv=shape.op==Op::NvProject;
                    let s=Shape {op:if nv {Op::NvRows} else {Op::Fp8Rows},tokens:m as u32,..*shape};
                    let lut=self.lut.as_ref().ok_or("expert decode table is absent")?;
                    // FP8 reads BF16-rounded lanes. Round once for all weight rows.
                    let source=if nv {b.expert_group(&[(Shape {op:Op::QuantizeRows,..s},None,0)],input,cols,lut)?} else {let r=b.round_bf16_lanes(input)?;let mut copy=b.vector(r.len())?;b.copy_prefix(r.vector(),&mut copy,r.len())?;copy};
                    (b.expert_group(&[(s,Some((weight,scales)),0)],&source,if nv {cols/16*17} else {cols},lut)?,true,if nv {2} else {3})
                },
            };
            let bytes=match &matrix.storage {Storage::Mx(_)=>matrix.rows.count*(matrix.cols+matrix.cols/32),Storage::Plain {weight,..}=>weight.len()*4,Storage::Packed {weight,scales,..}=>(weight.len()+scales.len())*4};
            self.cost.settle(b)?;self.cost.record(||if name=="lm_head.weight" {"projection_head".into()} else {format!("projection_{class}")},stamp,b,bytes);
            if !matrix.rows.split || self.collecting.get() {
                if !row_major || m==1 {return Ok(output);}
                let stamp=self.cost.start(b);let out=transpose(&output,local,m)?;self.cost.record(||"rows_transpose".into(),stamp,b,0);return Ok(out);
            }
            let rows=matrix.rows;let source=if row_major || m==1 {output} else {transpose(&output,m,local)?};
            if let Some(gathered)=self.gather_small(b,&source)? {drop(source);
                // [total rows, m] in rank order. One token is token-major as it is.
                if m==1 {return Ok(gathered);}
                let stamp=self.cost.start(b);let out=transpose(&gathered,rows.total,m)?;self.cost.record(||"exchange_upload".into(),stamp,b,0);return Ok(out);
            }
            self.gather_mapped(b,&source)?;drop(source);let stamp=self.cost.start(b);
            let exchange=self.exchange.as_ref().ok_or("prompt exchange is absent")?;
            let mut s=RowShape::new(Row::Transpose,1).ok_or("prompt transpose shape")?;
            s.rows=u32::try_from(rows.total).map_err(|_|"prompt transpose exceeds ABI")?;s.cols=u32::try_from(m).map_err(|_|"prompt transpose exceeds ABI")?;s.first=1;
            // One token still copies out: the exchange buffer is reused by the next gather.
            let out=b.prompt_operation(s,[Some(exchange),None,None,None,None],None)?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);Ok(out)
        }
        /// One layer's selected experts in whole-layer launches, one gather, one placement.
        /// None when a matrix is not rank-split NVFP4. Each element keeps its fold.
        pub fn apply_experts(&mut self,b:&Backend<'d>,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,input:&DeviceVector<'_>,places:&[f32])->Result<Option<DeviceVector<'d>>> {
            use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
            if first.is_empty() || !places.len().is_multiple_of(3) {return Err("prompt expert group is empty".into());}
            let info=|list:&[(&str,usize)],this:&Self|->Result<Option<(usize,usize,usize)>> {
                let (mut local,mut cols,mut total)=(0,0,0);
                for (index,(name,_)) in list.iter().enumerate() {
                    let matrix=this.matrices.get(*name).ok_or_else(||format!("projection is not resident: {name}"))?;
                    let Storage::Packed {shape,..}=&matrix.storage else {return Ok(None);};
                    if shape.op!=Op::NvProject || !matrix.rows.split {return Ok(None);}
                    if index>0 && (matrix.rows.count!=local || matrix.cols!=cols || matrix.rows.total!=total) {return Ok(None);}
                    local=matrix.rows.count;cols=matrix.cols;total=matrix.rows.total;
                }Ok(Some((local,cols,total)))};
            let Some((local,cols,total))=info(first,self)? else {return Ok(None);};
            if let Some((list,_))=second&& info(list,self)?!=Some((local,cols,total)) {return Ok(None);}
            if self.lut.is_none() {self.lut=Some(b.upload_vector(&spark_gpu_probe::expert_ops::decode_lut())?);}
            let stamp=self.cost.start(b);
            let n=first.iter().map(|v|v.1).sum::<usize>();
            if input.len()!=n.checked_mul(cols).ok_or("prompt expert input overflow")? || input.is_segmented() {return Err("prompt expert input extent differs".into());}
            let stride=cols/16*17;
            let packed_shape=|name:&str,this:&Self|->Result<Shape> {match &this.matrices.get(name).ok_or("expert matrix is absent")?.storage {Storage::Packed {shape,..}=>Ok(*shape),_=>Err("expert storage differs".to_string())}};
            let lut=self.lut.as_ref().ok_or("expert decode table is absent")?;
            // Quantize each expert's rows at its own calibrated input scale.
            let quantize=|list:&[(&str,usize)],this:&Self|->Result<DeviceVector<'d>> {
                let mut q=Vec::with_capacity(list.len());let mut row=0;
                for (name,count) in list {q.push((Shape {op:Op::QuantizeRows,tokens:*count as u32,..packed_shape(name,this)?},None,row));row+=count;}
                b.expert_group(&q,input,cols,lut)};
            let project=|list:&[(&str,usize)],packed:&DeviceVector<'d>,this:&Self|->Result<DeviceVector<'d>> {
                let mut p=Vec::with_capacity(list.len());let mut row=0;
                for (name,count) in list {let Storage::Packed {shape,weight,scales}=&this.matrices[*name].storage else {return Err("expert storage differs".into());};
                    p.push((Shape {op:Op::NvRows,tokens:*count as u32,..*shape},Some((weight,scales)),row));row+=count;}
                b.expert_group(&p,packed,stride,lut)};
            let packed=quantize(first,self)?;let a=project(first,&packed,self)?;
            let output=if let Some((list,bound))=second {
                // Equal input scale bits give equal quantized rows, so gate and up share them.
                let mut same=list.len()==first.len();
                if same {for (x,y) in first.iter().zip(list) {same&=packed_shape(x.0,self)?.activation.to_bits()==packed_shape(y.0,self)?.activation.to_bits() && x.1==y.1;}}
                let c=if same {project(list,&packed,self)?} else {let other=quantize(list,self)?;project(list,&other,self)?};
                let s=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::SwiGlu,heads:0,dim:0,taps:0,rows:a.len() as u32,cols:0,epsilon:0.0,bound};
                b.causal_operation(s,[Some(&a),Some(&c),None,None],None)?
            } else {a};drop(packed);
            self.cost.settle(b)?;self.cost.record(||"projection_2".into(),stamp,b,0);
            let parts=total/local;let words=output.len();
            let gathered=self.gather_small(b,&output)?;if gathered.is_none() {self.gather_mapped(b,&output)?;}drop(output);let stamp=self.cost.start(b);
            let exchange=match &gathered {Some(blocks)=>blocks,None=>self.exchange.as_ref().ok_or("prompt exchange is absent")?};
            // A table that is a real expert grouping also carries its slot to row inverse, and the placing runs in shared-memory tiles.
            let tiled=spark_gpu_probe::prompt_ops::place_table(places);
            let table=b.upload_vector(tiled.as_deref().unwrap_or(places))?;
            let mut s=RowShape::new(if tiled.is_some() {Row::ExpertPlaceTile} else {Row::ExpertPlace},1).ok_or("prompt expert place shape")?;
            s.rows=u32::try_from(places.len()/3).map_err(|_|"prompt expert rows exceed ABI")?;s.cols=u32::try_from(total).map_err(|_|"prompt expert rows exceed ABI")?;
            s.dim=u32::try_from(local).map_err(|_|"prompt expert rows exceed ABI")?;s.taps=parts as u32;s.slots=u32::try_from(words).map_err(|_|"prompt expert rows exceed ABI")?;s.first=1;
            let out=b.prompt_operation(s,[Some(exchange),Some(&table),None,None,None],None)?;
            self.cost.settle(b)?;self.cost.record(||"exchange_upload".into(),stamp,b,0);Ok(Some(out))
        }
        pub fn check_exchange(&self)->Result<()> {
            self.mailbox.map_or(Ok(()),|m|m.require_success())?;self.mhc_host.as_ref().map_or(Ok(()),|link|link.require_success())
        }
        pub fn report(&mut self) {self.cost.report();
            for (name,timing) in ["mxfp8","plain","nvfp4","fp8_block","mixed"].into_iter().zip(&mut self.timings) {
                eprintln!("RESIDENT COST rank={} storage={name} calls={} compute_ms={:.6} gather_ms={:.6}",self.topology.rank,timing[0],timing[1] as f64/1e6,timing[2] as f64/1e6);*timing=[0;3];
            }
        }
    }
    /// The matrices the plain single-token step reads by itself (not inside the mixed batch): router gate, kv_b, indexer wq_b, f_b and g_b.
    fn bf16_copy_wanted(name:&str,rows:usize,cols:usize)->bool {
        spark_gpu_probe::plain_admission::bf16_listed(rows,cols)
            && cols.is_multiple_of(8) && ["mlp.gate.weight","self_attn.kv_b_proj.weight","indexer.wq_b.weight","self_attn.f_b_proj.weight","self_attn.g_b_proj.weight"].iter().any(|n|name.ends_with(n))
    }
    impl<'b,'d> Matrix<'b,'d> {
        fn read(c:&CheckpointAdmission,name:&str,t:Topology)->Result<Prepared> {
            let view=c.headers().weights().get(name).ok_or("missing resident weight")?;let w=view.logical();let rows=Rows::new(&w.shape,w.storage,t)?;
            let cols=usize::try_from(w.shape[1]).map_err(|_|"resident columns exceed address space")?;
            let (upload,timings)=if w.storage==WeightStorage::Mxfp8 {(None,Default::default())} else {
                let (upload,timings)=c.upload_rows_timed(name,rows.start as u64..(rows.start+rows.count) as u64).map_err(|error|error.to_string())?;(Some(upload),timings)
            };
            Ok(Prepared {rows,cols,upload,timings})
        }
        fn load(c:&CheckpointAdmission,name:&str,t:Topology,b:&'b Backend<'d>,prepared:Prepared,load:&mut LoadTimings)->Result<Self> {
            let Prepared {rows,cols,upload,timings}=prepared;
            load.payload.allocate_ns+=timings.allocate_ns;load.payload.read_ns+=timings.read_ns;load.payload.unpack_ns+=timings.unpack_ns;
            load.payload.validation_ns+=timings.validation_ns;load.payload.bytes+=timings.bytes;
            let storage=if let Some(u)=upload {
                match u.storage {
                    WeightStorage::Mxfp8=>return Err("resident MXFP8 payload uses its existing arena loader".into()),
                    _=>Self::upload(name,rows,cols,u,b,load)?,
                }
            } else {
                let started=std::time::Instant::now();let topology=if rows.split {t} else {Topology {rank:0,parts:1}};
                let storage=Storage::Mx(Mxfp8Projection::matrix_rows(c,name,WeightType::F32,topology)?.load_dynamic_positive_zero(b)?);
                load.mxfp8_ns+=started.elapsed().as_nanos();storage
            };Ok(Self {rows,cols,storage})
        }
        fn upload(name:&str,rows:Rows,cols:usize,u:engine_model::plan::admission::UploadWeight,b:&'b Backend<'d>,load:&mut LoadTimings)->Result<Storage<'b,'d>> {
            Ok(match u.storage {
                    WeightStorage::Plain=>{
                        let mut s=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
                        s.rows=rows.count.try_into().map_err(|_|"resident rows exceed ABI")?;s.cols=cols.try_into().map_err(|_|"resident columns exceed ABI")?;
                        let width=u.dtype.bytes() as usize;let start=0;let end=rows.count*cols*width;
                        let convert=std::time::Instant::now();let values=match u.dtype {Dtype::Bf16=>u.weight[start..end].as_chunks::<2>().0.iter().map(|v|f32::from_bits(u32::from(u16::from_le_bytes(*v))<<16)).collect::<Vec<_>>(),
                            Dtype::F32=>u.weight[start..end].as_chunks::<4>().0.iter().map(|v|f32::from_le_bytes(*v)).collect(),_=>return Err("resident plain precision is unsupported".into())};
                        if values.iter().any(|v|!v.is_finite()) {return Err("resident plain weight is not finite".into());}
                        load.convert_ns+=convert.elapsed().as_nanos();let upload=std::time::Instant::now();let weight=b.upload_vector(&values)?;
                        load.upload_ns+=upload.elapsed().as_nanos();
                        // MEASURING FORM (on in code, no switch): the checkpoint's own BF16 bytes, packed two to a word, beside the FP32 copy. The
                        // single-token projections of the step that read these matrices use project_bf16_ordered; every other reader keeps the FP32 copy.
                        let bf16=if u.dtype==Dtype::Bf16 && s.op==spark_gpu_probe::causal_ops::Op::Project && bf16_copy_wanted(name,s.rows as usize,s.cols as usize) {
                            let words=u.weight[start..end].as_chunks::<4>().0.iter().map(|v|f32::from_bits(u32::from_le_bytes(*v))).collect::<Vec<_>>();
                            let packed=b.upload_vector(&words)?;BF16_COPY_BYTES.fetch_add(packed.len()*4,std::sync::atomic::Ordering::Relaxed);BF16_COPY_MATRICES.fetch_add(1,std::sync::atomic::Ordering::Relaxed);Some(packed)
                        } else {None};
                        Storage::Plain {shape:s,weight,bf16}
                    },
                    WeightStorage::Nvfp4 {..}|WeightStorage::Fp8Block {..}=>{
                        let mut s=Shape {op:Op::NvProject,tokens:1,rows:rows.count.try_into().map_err(|_|"resident rows exceed ABI")?,cols:cols.try_into().map_err(|_|"resident columns exceed ABI")?,block_rows:1,block_cols:16,
                            global:u.global_scale.unwrap_or(1.0),activation:u.input_scale.unwrap_or(1.0),bound:0.0};
                        let (stride,scale_start,scale_len)=match u.storage {WeightStorage::Nvfp4 {..}=>{
                            if u.global_scale.is_none() || u.input_scale.is_none() {return Err("resident calibrated scales are absent".into());}
                            (cols/2,0,rows.count*(cols/16))},
                            WeightStorage::Fp8Block {block}=>{s.op=Op::Fp8Serial;s.block_rows=block[0].try_into().map_err(|_|"scale row block exceeds ABI")?;s.block_cols=block[1].try_into().map_err(|_|"scale column block exceeds ABI")?;
                                let grid=cols.div_ceil(block[1] as usize);(cols,0,rows.count.div_ceil(block[0] as usize)*grid*4)},_=>unreachable!()};
                        s.lengths().ok_or("resident packed shape exceeds ABI")?;
                        let upload=std::time::Instant::now();let storage=Storage::Packed {shape:s,weight:raw(b,&u.weight[..rows.count*stride])?,scales:raw(b,&u.scales[scale_start..scale_start+scale_len])?};
                        load.upload_ns+=upload.elapsed().as_nanos();storage
                    },_=>return Err("resident storage is unsupported".into()),
            })
        }
    }
}
#[cfg(target_os="linux")]
pub use device::{Weights,gather_stage,gather_argmax};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_argmax_uses_global_origins_and_rejects_every_invalid_part() {
        let rows=[Rows {start:4096,count:4096,total:8192,split:true},Rows {start:0,count:4096,total:8192,split:true}];
        assert_eq!(merge_argmax(&[7.0,0.0,1.0,7.0,4095.0,1.0],&rows).unwrap(),4095);
        assert_eq!(merge_argmax(&[7.0,0.0,1.0,6.0,4095.0,1.0],&rows).unwrap(),4096);
        assert_eq!(merge_argmax(&[-0.0,0.0,1.0,0.0,2.0,1.0],&rows).unwrap(),2);
        assert_eq!(merge_argmax(&[-1.0,3.0,1.0,-2.0,0.0,1.0],&rows).unwrap(),4099);
        for invalid in [0.0,-1.0,f32::NAN,f32::INFINITY] {
            assert!(merge_argmax(&[7.0,0.0,1.0,-2.0,0.0,invalid],&rows).is_err());
        }
        for invalid in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
            assert!(merge_argmax(&[7.0,0.0,1.0,invalid,0.0,1.0],&rows).is_err());
        }
        for id in [-1.0,0.5,4096.0,f32::NAN,f32::INFINITY] {assert!(merge_argmax(&[7.0,id,1.0,6.0,0.0,1.0],&rows).is_err());}
        assert!(merge_argmax(&[],&[]).is_err());assert!(merge_argmax(&[7.0,0.0,1.0],&rows).is_err());
    }
    #[test]
    fn all_stored_bf16_patterns_are_fixed_points_of_the_projection_cast() {
        for bits in 0..=u16::MAX {let value=f32::from_bits(u32::from(bits)<<16);
            assert_eq!(spark_gpu_probe::expert_ops::round(value).to_bits(),value.to_bits());}
    }
    #[test]
    fn mixed_rows_restore_each_matrix_and_do_not_repeat_replicated_rows() {
        for chunks in 1..=7 {let rows=[Rows {start:0,count:3,total:3*chunks,split:chunks>1},Rows {start:0,count:1,total:1,split:false},Rows {start:0,count:5,total:5*chunks,split:chunks>1}];
            let packed=(0..chunks).flat_map(|rank|(0..9).map(move |lane|(rank*100+lane) as f32)).collect::<Vec<_>>();
            let output=gathered_mixed(&packed,&rows).unwrap();
            assert_eq!(output[0],(0..chunks).flat_map(|rank|(0..3).map(move |lane|(rank*100+lane) as f32)).collect::<Vec<_>>());
            assert_eq!(output[1],[3.0]);
            assert_eq!(output[2],(0..chunks).flat_map(|rank|(4..9).map(move |lane|(rank*100+lane) as f32)).collect::<Vec<_>>());
            assert!(gathered_mixed(&packed[..packed.len()-1],&rows).is_err());
        }
        assert!(gathered_mixed(&[],&[]).is_err());
        assert!(gathered_mixed(&[],&[Rows {start:0,count:2,total:5,split:true}]).is_err());
        assert!(gathered_mixed(&[],&[Rows {start:0,count:0,total:5,split:true}]).is_err());
        assert!(gathered_mixed(&[],&[Rows {start:0,count:1,total:2,split:true},Rows {start:0,count:1,total:3,split:true}]).is_err());
    }
    #[test]
    fn independent_jobs_gather_rank_major_then_matrix_rows() {
        for parts in 1..=7 {for jobs in [1,2,8,16] {for count in [1,5,17] {
            let rows=Rows {start:0,count,total:count*parts,split:parts>1};
            let packed=(0..parts).flat_map(|rank|(0..jobs).flat_map(move |job|(0..count).map(move |row|(rank*10000+job*100+row) as f32))).collect::<Vec<_>>();
            let output=gathered_batch(&packed,rows.count,rows.total,jobs).unwrap();
            for (job,values) in output.iter().enumerate() {assert_eq!(*values,(0..parts).flat_map(|rank|(0..count).map(move |row|(rank*10000+job*100+row) as f32)).collect::<Vec<_>>());}
            assert!(gathered_batch(&packed[..packed.len()-1],rows.count,rows.total,jobs).is_err());
        }}}
        let rows=Rows {start:0,count:2,total:2,split:false};
        assert_eq!(gathered_batch(&[-0.0,1.0,2.0,-0.0],rows.count,rows.total,2).unwrap().into_iter().flatten().map(f32::to_bits).collect::<Vec<_>>(),[-0.0,1.0,2.0,-0.0].map(f32::to_bits));
        assert!(gathered_batch(&[],rows.count,rows.total,0).is_err());
        assert!(gathered_batch(&[],usize::MAX,usize::MAX,2).is_err());
        assert!(gathered_batch(&[],0,rows.total,1).is_err());
        assert!(gathered_batch(&[],2,5,1).is_err());
    }
    #[test]
    fn row_owners_retain_complete_scale_blocks_and_replicate_small_outputs() {
        for storage in [WeightStorage::Mxfp8,WeightStorage::Nvfp4 {encoding:engine_model::plan::storage::ScaleEncoding::Csf32,activation_scale:true},WeightStorage::Fp8Block {block:[128,128]}] {
            let a=Rows::new(&[1024,4096],storage,Topology {rank:0,parts:2}).unwrap();let b=Rows::new(&[1024,4096],storage,Topology {rank:1,parts:2}).unwrap();
            assert_eq!(a,Rows {start:0,count:512,total:1024,split:true});assert_eq!(b.start,a.count);assert_eq!(b.start+b.count,a.total);
        }
        assert!(!Rows::new(&[1,4096],WeightStorage::Plain,Topology {rank:1,parts:2}).unwrap().split);
        assert!(Rows::new(&[0,32],WeightStorage::Mxfp8,Topology {rank:0,parts:2}).is_err());
    }
}
