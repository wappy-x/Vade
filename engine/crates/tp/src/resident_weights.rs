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
        ring:Option<Ring>,topology:Topology,identity:Option<usize>,cost:crate::resident_profile::Cost,timings:[[u128;3];4]}
    fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {
        let mut out=b.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;b.write_vector_bytes(&mut out,bytes)?;b.synchronize()?;Ok(out)
    }
    impl<'b,'d> Weights<'b,'d> {
        pub fn new(ring:Option<Ring>,cost:bool)->Self {
            let topology=ring.as_ref().map_or(Topology {rank:0,parts:1},|r|Topology {rank:r.rank() as u32,parts:r.world() as u32});
            Self {matrices:BTreeMap::new(),embedding:None,ring,topology,identity:None,cost:crate::resident_profile::Cost::new(cost),timings:[[0;3];4]}
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
                Storage::Mx(p)=>{let pass=p.project_vector(input)?;let v=pass.output().vector();let mut output=b.vector(v.len())?;b.copy_prefix(v,&mut output,v.len())?;b.synchronize()?;output},
                Storage::Plain {shape,weight}=>b.causal_operation(*shape,[Some(input),None,None,None],Some(weight))?,
                Storage::Packed {shape,weight,scales}=>{
                    let packed=if shape.op==Op::NvProject {Some(b.expert_operation(Shape {op:Op::Quantize,..*shape},[Some(input),None,None,None],None,None)?)} else {None};
                    b.expert_operation(*shape,[Some(packed.as_ref().unwrap_or(input)),None,None,None],Some(weight),Some(scales))?
                },
            };
            let output=if round {let rounded=b.round_bf16_lanes(&output)?;let mut copy=b.vector(output.len())?;b.copy_prefix(rounded.vector(),&mut copy,output.len())?;b.synchronize()?;copy} else {output};
            let bytes=match &matrix.storage {Storage::Mx(_)=>matrix.rows.count*(matrix.cols+matrix.cols/32),Storage::Plain {weight,..}=>weight.len()*4,Storage::Packed {weight,scales,..}=>(weight.len()+scales.len())*4};
            self.cost.record(||if name=="lm_head.weight" {"projection_head".into()} else {format!("projection_{class}")},stamp,b,bytes);let compute=started.elapsed().as_nanos();
            if !matrix.rows.split {self.timings[class][0]+=1;self.timings[class][1]+=compute;return Ok(output);}
            let stamp=self.cost.start(b);let local=b.read_vector_values(&output)?;let mut bytes=vec![0;matrix.rows.total*4];
            for (dst,v) in bytes[matrix.rows.start*4..(matrix.rows.start+matrix.rows.count)*4].chunks_exact_mut(4).zip(local) {dst.copy_from_slice(&v.to_le_bytes());}
            self.cost.record(||"exchange_pack".into(),stamp,b,0);let stamp=self.cost.start(b);
            self.ring.as_mut().ok_or("resident row split has no group")?.all_gather(&mut bytes,4)?;
            self.cost.record(||"exchange_network".into(),stamp,b,0);let stamp=self.cost.start(b);
            let values=bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) {return Err("resident gathered projection is not finite".into());}
            let output=b.upload_vector(&values)?;b.synchronize()?;
            self.cost.record(||"exchange_upload".into(),stamp,b,0);self.timings[class][0]+=1;self.timings[class][1]+=compute;self.timings[class][2]+=started.elapsed().as_nanos()-compute;Ok(output)
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
        pub fn report(&mut self) {self.cost.report();
            for (name,timing) in ["mxfp8","plain","nvfp4","fp8_block"].into_iter().zip(&mut self.timings) {
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
                Storage::Mx(Mxfp8Projection::matrix_rows(c,name,WeightType::F32,topology)?.load_dynamic(b)?)
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
    fn row_owners_retain_complete_scale_blocks_and_replicate_small_outputs() {
        for storage in [WeightStorage::Mxfp8,WeightStorage::Nvfp4 {encoding:engine_model::plan::storage::ScaleEncoding::Csf32,activation_scale:true},WeightStorage::Fp8Block {block:[128,128]}] {
            let a=Rows::new(&[1024,4096],storage,Topology {rank:0,parts:2}).unwrap();let b=Rows::new(&[1024,4096],storage,Topology {rank:1,parts:2}).unwrap();
            assert_eq!(a,Rows {start:0,count:512,total:1024,split:true});assert_eq!(b.start,a.count);assert_eq!(b.start+b.count,a.total);
        }
        assert!(!Rows::new(&[1,4096],WeightStorage::Plain,Topology {rank:1,parts:2}).unwrap().split);
        assert!(Rows::new(&[0,32],WeightStorage::Mxfp8,Topology {rank:0,parts:2}).is_err());
    }
}
