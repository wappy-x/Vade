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
        let split=shape[0]%stride==0;
        if !split {return Ok(Self {start:0,count:total,total,split:false});}
        let owner=engine_model::plan::tensor_split::Ownership::equal(shape,0,unit,t.parts).map_err(|e|e.to_string())?;
        let interval=owner.owner(t.rank).map_err(|e|e.to_string())?.interval.clone();
        Ok(Self {start:interval.start as usize,count:(interval.end-interval.start) as usize,total,split:t.parts!=1})
    }
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

#[cfg(target_os="linux")]
mod device {
    use super::*;
    use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan}},safetensors::Dtype};
    use spark_gpu_probe::{driver::{Backend,DeviceVector},norm_boundary::WeightType,expert_ops::{Shape,Op}};
    use crate::{mxfp8_projection::Mxfp8Projection,dynamic_projection::ResidentDynamicProjection,transport::Ring};
    use std::collections::{BTreeMap,BTreeSet};
    enum Storage<'b,'d> {Mx(ResidentDynamicProjection<'b,'d>),Plain {shape:spark_gpu_probe::causal_ops::Shape,weight:DeviceVector<'d>},
        Packed {shape:Shape,weight:DeviceVector<'d>,scales:DeviceVector<'d>}}
    struct Matrix<'b,'d> {rows:Rows,cols:usize,storage:Storage<'b,'d>}
    pub struct Weights<'b,'d> {matrices:BTreeMap<String,Matrix<'b,'d>>,embedding:Option<spark_gpu_probe::driver::Matrix<'d>>,
        ring:Option<Ring>,mailbox:Option<&'b spark_gpu_probe::driver::TpMailbox<'d>>,sequence:u64,topology:Topology,identity:Option<usize>,cost:crate::resident_profile::Cost,timings:[[u128;3];5],lut:Option<DeviceVector<'d>>,exchange:Option<DeviceVector<'d>>}
    fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {
        let mut out=b.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;b.write_vector_bytes(&mut out,bytes)?;b.synchronize()?;Ok(out)
    }
    impl<'b,'d> Weights<'b,'d> {
        pub fn new(ring:Option<Ring>,cost:bool)->Self {
            let topology=ring.as_ref().map_or(Topology {rank:0,parts:1},|r|Topology {rank:r.rank() as u32,parts:r.world() as u32});
            Self {matrices:BTreeMap::new(),embedding:None,ring,mailbox:None,sequence:0,topology,identity:None,cost:crate::resident_profile::Cost::new(cost),timings:[[0;3];5],lut:None,exchange:None}
        }
        pub fn attach_mailbox(&mut self,mailbox:&'b spark_gpu_probe::driver::TpMailbox<'d>)->Result<()> {
            let physical=Topology {rank:self.topology.rank,parts:mailbox.layout().world.try_into().map_err(|_|"mailbox topology exceeds ABI")?};
            let compatible=self.topology.eq(&physical);
            if !compatible || self.identity.is_some() || self.mailbox.is_some() {
                return Err("resident device exchange requires a fresh pair owner".into());
            }
            mailbox.claim_copy_group()?;self.mailbox=Some(mailbox);Ok(())
        }
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
                    WeightStorage::Plain=>rows.count.checked_mul(cols).and_then(|n|n.checked_mul(4)),
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
            let started=std::time::Instant::now();
            for (index,name) in names.into_iter().enumerate() {
                let matrix=Matrix::load(c,&name,self.topology,b).map_err(|e|format!("{name}: {e}"))?;self.matrices.insert(name,matrix);
                if index%256==0 {eprintln!("RESIDENT LOAD rank={} matrices={} allocated_bytes={}",self.topology.rank,index+1,b.allocated_bytes());
                    // Loading can exceed the normal exchange deadline. Keep peers
                    // together at bounded batches before any token transaction.
                    if let Some(ring)=self.ring.as_mut() {ring.barrier()?;}
                }
            }
            let file=c.component_file(embedding_name).map_err(|e|e.to_string())?;let tensor=&file.header().tensors[embedding_name];
            let offset=file.header().data_start.checked_add(tensor.start).ok_or("resident embedding file offset overflow")?;
            let ranges=[spark_gpu_probe::load::WeightRange {file_offset:offset,destination:0,bytes:embedding_bytes,row_bytes:0,row_stride:0}];
            let arena=b.load_weight_arena(embedding_bytes,&[spark_gpu_probe::load::WeightFile {file:file.file(),ranges:&ranges}])?;
            self.embedding=Some(b.weight_matrix(&arena,0,d.vocab as usize,d.hidden as usize,spark_gpu_probe::load::WeightDtype::Bf16)?);
            b.synchronize()?;if let Some(r)=self.ring.as_mut() {r.barrier()?;}
            self.identity=Some(identity);
            eprintln!("RESIDENT READY rank={} parts={} matrices={} matrix_bytes={bytes} allocated_bytes={} load_seconds={:.6}",self.topology.rank,self.topology.parts,self.matrices.len(),b.allocated_bytes(),started.elapsed().as_secs_f64());Ok(())
        }
        pub fn embedding(&self,b:&Backend<'d>,token:u32,streams:usize)->Result<DeviceVector<'d>> {
            let table=self.embedding.as_ref().ok_or("embedding is not resident")?;
            if token as usize>=table.rows() || streams==0 {return Err("resident embedding ID or stream count differs".into());}
            let ids=b.upload_vector(&vec![token as f32;streams])?;let mut out=b.vector(streams.checked_mul(table.cols()).ok_or("resident embedding output overflow")?)?;
            let launched=b.batch_embedding_device(table,&ids,&mut out);let completed=b.synchronize();launched.and(completed)?;Ok(out)
        }
        pub fn embedding_rows(&self,b:&Backend<'d>,tokens:&[u32],streams:usize)->Result<DeviceVector<'d>> {
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
            let minimum=bytes.chunks_exact(8).map(|v|u64::from_le_bytes(v.try_into().unwrap())).min().ok_or("empty resident budget agreement")?;
            usize::try_from(minimum).map_err(|_|"resident shared budget exceeds address space".into())
        }
        pub fn apply(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,round:bool)->Result<DeviceVector<'d>> {
            let stamp=self.cost.start(b);let started=std::time::Instant::now();
            let matrix=self.matrices.get(name).ok_or_else(||format!("projection is not resident: {name}"))?;
            if input.len()!=matrix.cols || input.is_segmented() {return Err("resident projection feature extent differs".into());}
            let class=match &matrix.storage {Storage::Mx(_)=>0,Storage::Plain {..}=>1,Storage::Packed {shape,..}=>if shape.op==Op::NvProject {2} else {3}};
            let output=match &matrix.storage {
                Storage::Mx(p)=>p.project_vector(input)?.into_output().into_vector(),
                Storage::Plain {shape,weight}=>b.causal_operation(*shape,[Some(input),None,None,None],Some(weight))?,
                Storage::Packed {shape,weight,scales}=>{
                    let packed=if shape.op==Op::NvProject {Some(b.expert_operation(Shape {op:Op::Quantize,..*shape},[Some(input),None,None,None],None,None)?)} else {None};
                    b.expert_operation(*shape,[Some(packed.as_ref().unwrap_or(input)),None,None,None],Some(weight),Some(scales))?
                },
            };
            // Packed projection kernels and the typed MX output already store BF16
            // lanes. A second cast is an identity for every stored BF16 pattern.
            // Plain ordered projections still require their explicit boundary.
            let rounded=matches!(&matrix.storage,Storage::Mx(_)|Storage::Packed {shape:Shape {op:Op::NvProject|Op::Fp8Project,..},..});
            let output=if round && !rounded {b.round_bf16_lanes(&output)?.into_vector()} else {output};
            let bytes=match &matrix.storage {Storage::Mx(_)=>matrix.rows.count*(matrix.cols+matrix.cols/32),Storage::Plain {weight,..}=>weight.len()*4,Storage::Packed {weight,scales,..}=>(weight.len()+scales.len())*4};
            self.cost.record(||if name=="lm_head.weight" {"projection_head".into()} else {format!("projection_{class}")},stamp,b,bytes);let compute=started.elapsed().as_nanos();
            if !matrix.rows.split {self.timings[class][0]+=1;self.timings[class][1]+=compute;return Ok(output);}
            if let Some(mailbox)=self.mailbox {
                let stamp=self.cost.start(b);
                let sequence=self.sequence.checked_add(1).filter(|&n|n<=u32::MAX as u64).ok_or("resident mailbox sequence exhausted")?;
                if mailbox.failed() {b.synchronize()?;return Err("resident device exchange failed".into());}
                let total=matrix.rows.total;let count=matrix.rows.count;let own=matrix.rows.start;
                // A failed take writes no peer rows. Initialize them before any queued consumer.
                let mut gathered=b.vector(total)?;b.zero_vector(&mut gathered)?;b.copy_range(&output,&mut gathered,0,own,count)?;
                b.tp_mailbox_put_finite(mailbox,&gathered,1,total,count,own,sequence)?;
                let peer=total.checked_sub(own+count).ok_or("resident peer interval overflow")?;
                b.tp_mailbox_take(mailbox,&mut gathered,1,total,count,peer,sequence,10_000_000_000)?;
                self.sequence=sequence;
                self.cost.record(||"exchange_device".into(),stamp,b,0);
                self.timings[class][0]+=1;self.timings[class][1]+=compute;self.timings[class][2]+=started.elapsed().as_nanos()-compute;
                return Ok(gathered);
            }
            let stamp=self.cost.start(b);let local=b.read_vector_values(&output)?;
let mut bytes=vec![0;matrix.rows.total*4];
            for (dst,v) in bytes[matrix.rows.start*4..(matrix.rows.start+matrix.rows.count)*4].chunks_exact_mut(4).zip(local) {dst.copy_from_slice(&v.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);let stamp=self.cost.start(b);
            self.ring.as_mut().ok_or("resident row split has no group")?.all_gather(&mut bytes,4)?;
            self.cost.record(||"exchange_network".into(),stamp,b,0);let stamp=self.cost.start(b);
            let values=bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) {return Err("resident gathered projection is not finite".into());}
            let output=b.upload_vector(&values)?;b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);self.timings[class][0]+=1;self.timings[class][1]+=compute;self.timings[class][2]+=started.elapsed().as_nanos()-compute;Ok(output)
        }
        pub fn apply_batch(&mut self,b:&Backend<'d>,input:&[(&str,&DeviceVector<'_>)],round:bool)->Result<Option<Vec<DeviceVector<'d>>>> {
            if input.is_empty() {return Err("resident projection batch is empty".into());}
            let matrices=input.iter().map(|(name,_)|self.matrices.get(*name).ok_or_else(||format!("projection is not resident: {name}"))).collect::<Result<Vec<_>>>()?;
            if input.len()<=9 && matrices.iter().all(|matrix|matches!(&matrix.storage,Storage::Mx(_)|Storage::Plain {..}) && matrix.cols%32==0) {
                return self.apply_mixed_batch(b,input,round);
            }
            let first=matrices[0];
            if matrices.iter().any(|matrix|matrix.rows!=first.rows || matrix.cols!=first.cols || !matches!(&matrix.storage,Storage::Packed {shape,..} if shape.op==Op::NvProject)) {return Ok(None);}
            if input.iter().any(|(_,value)|value.len()!=first.cols || value.is_segmented()) {return Err("resident batch feature extent differs".into());}
            let stamp=self.cost.start(b);let started=std::time::Instant::now();
            let mut quantized=Vec::new();let mut indices=Vec::new();let mut weight_bytes=0usize;
            for (matrix,(_,value)) in matrices.iter().zip(input) {
                let Storage::Packed {shape,weight,scales}=&matrix.storage else {unreachable!()};
                let key=(value.device_address(),matrix.cols,shape.activation.to_bits());
                let index=if let Some(index)=quantized.iter().position(|(prior,_)|*prior==key) {index} else {
                    let packed=b.expert_operation(Shape {op:Op::Quantize,..*shape},[Some(value),None,None,None],None,None)?;
                    quantized.push((key,packed));quantized.len()-1
                };indices.push(index);weight_bytes+=(weight.len()+scales.len())*4;
            }
            let jobs=matrices.iter().zip(&indices).map(|(matrix,index)| {let Storage::Packed {shape,weight,scales}=&matrix.storage else {unreachable!()};
                (*shape,&quantized[*index].1,weight,scales)
            }).collect::<Vec<_>>();
            let output=b.expert_projection_batch(&jobs)?;
            let output=if round {let rounded=b.round_bf16_lanes(&output)?;let mut copy=b.vector(output.len())?;b.copy_prefix(rounded.vector(),&mut copy,output.len())?;b.synchronize()?;copy} else {output};
            self.cost.record(||"projection_2".into(),stamp,b,weight_bytes);let compute=started.elapsed().as_nanos();let stamp=self.cost.start(b);
            let local=b.read_vector_values(&output)?;let ranks=if first.rows.split {self.topology.parts as usize} else {1};
            let mut bytes=vec![0;local.len().checked_mul(ranks).and_then(|words|words.checked_mul(4)).ok_or("resident batch exchange overflow")?];
            let start=if first.rows.split {self.topology.rank as usize*local.len()*4} else {0};
            for (destination,value) in bytes[start..start+local.len()*4].chunks_exact_mut(4).zip(local) {destination.copy_from_slice(&value.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);
            if first.rows.split {let stamp=self.cost.start(b);self.ring.as_mut().ok_or("resident batch has no row group")?.all_gather(&mut bytes,4)?;self.cost.record(||"exchange_network".into(),stamp,b,0);}
            let stamp=self.cost.start(b);let values=bytes.chunks_exact(4).map(|word|f32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
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
            let local=b.read_vector_values(&output)?;let split=matrices.iter().any(|matrix|matrix.rows.split);let ranks=if split {self.topology.parts as usize} else {1};
            let mut bytes=vec![0;local.len().checked_mul(ranks).and_then(|words|words.checked_mul(4)).ok_or("resident mixed batch exchange overflow")?];
            let start=if split {self.topology.rank as usize*local.len()*4} else {0};
            for (destination,value) in bytes[start..start+local.len()*4].chunks_exact_mut(4).zip(local) {destination.copy_from_slice(&value.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);
            if split {let stamp=self.cost.start(b);self.ring.as_mut().ok_or("resident mixed batch has no row group")?.all_gather(&mut bytes,4)?;self.cost.record(||"exchange_network".into(),stamp,b,0);}
            let stamp=self.cost.start(b);let values=bytes.chunks_exact(4).map(|word|f32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
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
                Storage::Plain {shape,weight}=>(b.causal_operation(spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::ProjectRows,heads:m as u32,..*shape},[Some(&joined),None,None,None],Some(weight))?,1),
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
            for (dst,value) in bytes[matrix.rows.start*m*4..(matrix.rows.start+matrix.rows.count)*m*4].chunks_exact_mut(4).zip(local) {dst.copy_from_slice(&value.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);
            if matrix.rows.split {let stamp=self.cost.start(b);self.ring.as_mut().ok_or("resident row split has no group")?.all_gather(&mut bytes,m*4)?;self.cost.record(||"exchange_network".into(),stamp,b,0);}
            let stamp=self.cost.start(b);let values=bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) {return Err("resident prompt projection is not finite".into());}
            let mut result=Vec::new();for token in 0..m {let row=(0..matrix.rows.total).map(|column|values[column*m+token]).collect::<Vec<_>>();result.push(b.upload_vector(&row)?);}b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);Ok(result)
        }
        /// All ranks' equal local vectors, rank-major, in one mapped allocation. GB10 has
        /// one DRAM pool, so the wire writes it and later kernels read it, with no host
        /// copy. The allocation grows rarely and stays registered between transfers.
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
            crate::model::exchange_blocks(b,self.ring.as_mut().ok_or("resident row split has no group")?,exchange,parts,1,words)?;
            self.cost.record(||"exchange_network".into(),stamp,b,0);let stamp=self.cost.start(b);
            // A driver copy, so no host view of this allocation exists outside the link step.
            if b.read_vector_values(exchange)?.iter().any(|v|!v.is_finite()) {return Err("resident prompt projection is not finite".into());}
            self.cost.record(||"exchange_check".into(),stamp,b,0);Ok(())
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
                Ok(b.prompt_operation(s,[Some(x),None,None,None,None],None)?)};
            // Output in [local rows,m] when row-major, else [m,local rows].
            let (output,row_major,class)=match &matrix.storage {
                Storage::Mx(p)=>{let v=p.project_rows(input,m)?;let mut out=b.vector(m*local)?;b.copy_prefix(v.vector(),&mut out,m*local)?;(out,false,0)},
                Storage::Plain {shape,weight}=>{let mut s=RowShape::new(Row::ProjectRows,m).ok_or("prompt rows exceed ABI")?;s.rows=shape.rows;s.cols=shape.cols;
                    let out=b.prompt_operation(s,[Some(input),None,None,None,None],Some(weight))?;
                    (if round {let rounded=b.round_bf16_lanes(&out)?;let mut copy=b.vector(out.len())?;b.copy_prefix(rounded.vector(),&mut copy,out.len())?;copy} else {out},true,1)},
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
            if !matrix.rows.split {
                if !row_major || m==1 {return Ok(output);}
                let stamp=self.cost.start(b);let out=transpose(&output,local,m)?;self.cost.record(||"rows_transpose".into(),stamp,b,0);return Ok(out);
            }
            let rows=matrix.rows;let source=if row_major || m==1 {output} else {transpose(&output,m,local)?};
            self.gather_mapped(b,&source)?;drop(source);let stamp=self.cost.start(b);
            let exchange=self.exchange.as_ref().ok_or("prompt exchange is absent")?;
            let mut s=RowShape::new(Row::Transpose,1).ok_or("prompt transpose shape")?;
            s.rows=u32::try_from(rows.total).map_err(|_|"prompt transpose exceeds ABI")?;s.cols=u32::try_from(m).map_err(|_|"prompt transpose exceeds ABI")?;
            // One token still copies out: the exchange buffer is reused by the next gather.
            let out=b.prompt_operation(s,[Some(exchange),None,None,None,None],None)?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);Ok(out)
        }
        /// One layer's selected experts in whole-layer launches, one gather, one placement.
        /// None when a matrix is not rank-split NVFP4. Each element keeps its fold.
        pub fn apply_experts(&mut self,b:&Backend<'d>,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,input:&DeviceVector<'_>,places:&[f32])->Result<Option<DeviceVector<'d>>> {
            use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
            if first.is_empty() || places.len()%3!=0 {return Err("prompt expert group is empty".into());}
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
            if let Some((list,_))=second {if info(list,self)?!=Some((local,cols,total)) {return Ok(None);}}
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
                Ok(b.expert_group(&q,input,cols,lut)?)};
            let project=|list:&[(&str,usize)],packed:&DeviceVector<'d>,this:&Self|->Result<DeviceVector<'d>> {
                let mut p=Vec::with_capacity(list.len());let mut row=0;
                for (name,count) in list {let Storage::Packed {shape,weight,scales}=&this.matrices[*name].storage else {return Err("expert storage differs".into());};
                    p.push((Shape {op:Op::NvRows,tokens:*count as u32,..*shape},Some((weight,scales)),row));row+=count;}
                Ok(b.expert_group(&p,packed,stride,lut)?)};
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
            self.gather_mapped(b,&output)?;drop(output);let stamp=self.cost.start(b);
            let exchange=self.exchange.as_ref().ok_or("prompt exchange is absent")?;let table=b.upload_vector(places)?;
            let mut s=RowShape::new(Row::ExpertPlace,1).ok_or("prompt expert place shape")?;
            s.rows=u32::try_from(places.len()/3).map_err(|_|"prompt expert rows exceed ABI")?;s.cols=u32::try_from(total).map_err(|_|"prompt expert rows exceed ABI")?;
            s.dim=u32::try_from(local).map_err(|_|"prompt expert rows exceed ABI")?;s.taps=parts as u32;s.slots=u32::try_from(words).map_err(|_|"prompt expert rows exceed ABI")?;
            let out=b.prompt_operation(s,[Some(exchange),Some(&table),None,None,None],None)?;
            self.cost.settle(b)?;self.cost.record(||"exchange_upload".into(),stamp,b,0);Ok(Some(out))
        }
        pub fn check_exchange(&self)->Result<()> {
            self.mailbox.map_or(Ok(()),|m|m.require_success())
        }
        pub fn report(&mut self) {self.cost.report();
            for (name,timing) in ["mxfp8","plain","nvfp4","fp8_block","mixed"].into_iter().zip(&mut self.timings) {
                eprintln!("RESIDENT COST rank={} storage={name} calls={} compute_ms={:.6} gather_ms={:.6}",self.topology.rank,timing[0],timing[1] as f64/1e6,timing[2] as f64/1e6);*timing=[0;3];
            }
        }
    }
    impl<'b,'d> Matrix<'b,'d> {
        fn load(c:&CheckpointAdmission,name:&str,t:Topology,b:&'b Backend<'d>)->Result<Self> {
            let view=c.headers().weights().get(name).ok_or("missing resident weight")?;let w=view.logical();let rows=Rows::new(&w.shape,w.storage,t)?;
            let cols=usize::try_from(w.shape[1]).map_err(|_|"resident columns exceed address space")?;
            let storage=if w.storage==WeightStorage::Mxfp8 {
                let topology=if rows.split {t} else {Topology {rank:0,parts:1}};
                Storage::Mx(Mxfp8Projection::matrix_rows(c,name,WeightType::F32,topology)?.load_dynamic_positive_zero(b)?)
            } else {
                let u=c.upload(name).map_err(|e|e.to_string())?;
                match u.storage {
                    WeightStorage::Plain=>{
                        let mut s=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
                        s.rows=rows.count.try_into().map_err(|_|"resident rows exceed ABI")?;s.cols=cols.try_into().map_err(|_|"resident columns exceed ABI")?;
                        let width=u.dtype.bytes() as usize;let start=rows.start*cols*width;let end=start+rows.count*cols*width;
                        let values=match u.dtype {Dtype::Bf16=>u.weight[start..end].chunks_exact(2).map(|v|f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap()))<<16)).collect::<Vec<_>>(),
                            Dtype::F32=>u.weight[start..end].chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect(),_=>return Err("resident plain precision is unsupported".into())};
                        if values.iter().any(|v|!v.is_finite()) {return Err("resident plain weight is not finite".into());}
                        Storage::Plain {shape:s,weight:b.upload_vector(&values)?}
                    },
                    WeightStorage::Nvfp4 {..}|WeightStorage::Fp8Block {..}=>{
                        let mut s=Shape {op:Op::NvProject,tokens:1,rows:rows.count.try_into().map_err(|_|"resident rows exceed ABI")?,cols:cols.try_into().map_err(|_|"resident columns exceed ABI")?,block_rows:1,block_cols:16,
                            global:u.global_scale.unwrap_or(1.0),activation:u.input_scale.unwrap_or(1.0),bound:0.0};
                        let (stride,scale_start,scale_len)=match u.storage {WeightStorage::Nvfp4 {..}=>{
                            if u.global_scale.is_none() || u.input_scale.is_none() {return Err("resident calibrated scales are absent".into());}
                            (cols/2,rows.start*(cols/16),rows.count*(cols/16))},
                            WeightStorage::Fp8Block {block}=>{s.op=Op::Fp8Project;s.block_rows=block[0].try_into().map_err(|_|"scale row block exceeds ABI")?;s.block_cols=block[1].try_into().map_err(|_|"scale column block exceeds ABI")?;
                                let grid=cols.div_ceil(block[1] as usize);(cols,rows.start/(block[0] as usize)*grid*4,rows.count.div_ceil(block[0] as usize)*grid*4)},_=>unreachable!()};
                        s.lengths().ok_or("resident packed shape exceeds ABI")?;
                        Storage::Packed {shape:s,weight:raw(b,&u.weight[rows.start*stride..(rows.start+rows.count)*stride])?,scales:raw(b,&u.scales[scale_start..scale_start+scale_len])?}
                    },_=>return Err("resident storage is unsupported".into()),
                }
            };Ok(Self {rows,cols,storage})
        }
    }
}
#[cfg(target_os="linux")]
pub use device::Weights;

#[cfg(test)]
mod tests {
    use super::*;
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
