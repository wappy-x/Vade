//! Bind complete stored MXFP8 rows to a declared producer feature row.
//! This candidate does not qualify a checkpoint runtime or admit a full layer.
use crate::{Result,layer_driver::LayerBoundary};
use engine_model::{plan::{admission::{CheckpointAdmission,ComponentView},layer::{Kind,Topology,cast::Dtype as CastDtype},storage::WeightStorage},
    safetensors::{Dtype,TensorFile}};
use spark_gpu_probe::{mxfp8_projection::Shape,norm_boundary::WeightType};

fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
struct Component<'f> {file:&'f TensorFile,name:String,offset:u64,file_offset:u64,bytes:usize}
impl<'f> Component<'f> {
    fn bind(c:&ComponentView,checkpoint:&'f CheckpointAdmission,origin:u64,rows:u64,stride:u64)->Result<Self> {
        let file=checkpoint.component_file(&c.name).map_err(|e|e.to_string())?;
        let t=file.header().tensors.get(&c.name).ok_or("MXFP8 component is absent from its retained file")?;
        need(t.dtype==c.dtype && t.shape==c.shape && t.start==c.start && t.byte_len()==c.bytes,
            "MXFP8 retained header differs from its admitted component")?;
        let offset=origin.checked_mul(stride).ok_or("MXFP8 component offset overflow")?;
        let length=rows.checked_mul(stride).ok_or("MXFP8 component size overflow")?;
        need(offset.checked_add(length).is_some_and(|n|n<=c.bytes),"MXFP8 row interval exceeds its component")?;
        let file_offset=file.header().data_start.checked_add(t.start).and_then(|n|n.checked_add(offset)).ok_or("MXFP8 file offset overflow")?;
        need(file_offset.checked_add(length).is_some_and(|n|n<=file.header().file_bytes),"MXFP8 row interval exceeds its retained file")?;
        Ok(Self {file,name:c.name.clone(),offset,file_offset,bytes:usize::try_from(length).map_err(|_|"MXFP8 component exceeds address space")?})
    }
    fn read(&self)->Result<Vec<u8>> {
        let mut bytes=Vec::new();bytes.try_reserve_exact(self.bytes).map_err(|_|"MXFP8 row allocation failed")?;bytes.resize(self.bytes,0);
        self.file.read_range(&self.name,self.offset,&mut bytes).map_err(|e|e.to_string())?;Ok(bytes)
    }
}
pub struct Mxfp8Projection<'f> {
    shape:Shape,origin:u64,total_rows:u64,topology:Topology,input:String,output:String,gather:bool,
    weight:Component<'f>,scale:Component<'f>,
    /// The word function of this matrix (`next_params::MX_SCALAR` or `MX_TC`), fixed at the binding by the tensor's role
    /// (`word_of`). The device matrix and the host reference both take it from here, so every path of the matrix has one word.
    word:u32,
    /// Not the output head: a launch of this matrix on the scalar word is counted (`mx_scalar_layer`).
    layer:bool,
}
/// THE SWITCH of the MXFP8 word function. Every MXFP8 matrix of a layer (attention q, k, v, o; dense gate, up, down; every
/// other MXFP8 projection a layer binds) takes the tensor-core word of `docs/tc-dense-contract.md`, section MXFP8. The output
/// head keeps the scalar word: its kernels (`text_edges`, the head row forms) are not part of this switch.
pub const LAYER_WORD:u32=spark_gpu_probe::next_params::MX_TC;
/// The served proof lines of one rank: the MXFP8 word counters, every launch counter and the block FP8 word counters of this process so far, at the end
/// of a session's prompt (`at` = "prompt") and at the session's end ("end"), on every rank. The difference of the two is the
/// decode and verify phase of a lone session.
pub fn served_counts(rank:impl std::fmt::Display,session:impl std::fmt::Display,at:&str) {
    eprintln!("SERVED MX COUNTS rank={rank} session={session} at={at} {}",spark_gpu_probe::prompt_launches::mx_line());
    eprintln!("SERVED BF COUNTS rank={rank} session={session} at={at} {}",spark_gpu_probe::prompt_launches::line());
    // BLOCK FP8 on the tensor cores: which body the parts of every block FP8 launch took (`fp8_generic_layer` stays 0).
    eprintln!("SERVED FP8 COUNTS rank={rank} session={session} at={at} {}",spark_gpu_probe::prompt_launches::fp8_line());
}
fn is_layer(view:&engine_model::plan::admission::WeightView)->bool {view.binding().operation!=engine_model::plan::glm_index::TensorRole::Head}
fn word_of(view:&engine_model::plan::admission::WeightView)->u32 {if is_layer(view) {LAYER_WORD} else {spark_gpu_probe::next_params::MX_SCALAR}}
impl LayerBoundary<'_> {
    /// The actual norm parameter dtype determines its completed candidate output.
    /// Other producer dtypes follow the conditional layer boundary contract.
    /// A stored MXFP8 recipe alone cannot establish a runtime activation quantizer.
    pub fn prepare_mxfp8_projection_checkpoint<'f>(&self,step:usize,checkpoint:&'f CheckpointAdmission)->Result<Mxfp8Projection<'f>> {
        let p=self.program();let op=p.steps.get(step).ok_or("MXFP8 step is outside the layer program")?;
        need(op.kind==Kind::Project && op.inputs.len()==1 && op.outputs.len()==1 && op.weights.len()==1
            && self.casts().steps[step].outputs[0].dtype==CastDtype::Bf16,"MXFP8 candidate requires one ordinary BF16-output projection")?;
        let weights=self.weights(step)?;let binding=&weights[0];let view=binding.view();let w=view.logical();
        need(w.storage==WeightStorage::Mxfp8 && w.shape.len()==2,"projection requires declared MXFP8 matrix storage")?;
        need(checkpoint.headers().weights().get(&w.name)==Some(view),"MXFP8 checkpoint headers differ from the bound weight")?;
        let producer=p.steps[..step].iter().rposition(|s|s.outputs.iter().any(|b|b.name==op.inputs[0])).ok_or("MXFP8 input producer is absent")?;
        let norm=&p.steps[producer];
        let output=norm.outputs.iter().position(|b|b.name==op.inputs[0]).unwrap();
        need(norm.outputs[output].shape.iter().try_fold(1u64,|n,&v|n.checked_mul(v))==Some(w.shape[1]),
            "MXFP8 candidate requires a complete producer feature row")?;
        let input=if norm.kind==Kind::Norm {
            let norm_weights=self.weights(producer)?;
            need(norm_weights.len()==1,"MXFP8 norm producer requires one weight")?;
            let norm_view=norm_weights[0].view();
            need(norm_view.logical().storage==WeightStorage::Plain && norm_view.components().len()==1
                && checkpoint.headers().weights().get(&norm_view.logical().name)==Some(norm_view),"MXFP8 norm checkpoint headers differ")?;
            match norm_view.components()[0].dtype {Dtype::Bf16=>WeightType::Bf16,Dtype::F32=>WeightType::F32,
                _=>return Err("MXFP8 norm output has no bound stored-identity precision".into())}
        } else {match self.casts().steps[producer].outputs[output].dtype {
            CastDtype::Bf16=>WeightType::Bf16,CastDtype::F32=>WeightType::F32,
            _=>return Err("MXFP8 producer output precision is unresolved".into()),
        }};
        let (origin,rows)=binding.rows().map_or((0,w.shape[0]),|r| {let r=r.rows();(r.start,r.end-r.start)});
        let shape=Shape::new(usize::try_from(rows).map_err(|_|"MXFP8 rows exceed address space")?,
            usize::try_from(w.shape[1]).map_err(|_|"MXFP8 columns exceed address space")?,input)?;
        shape.arena_bytes()?;
        let c=view.components();
        need(c.len()==2 && c[0].name==w.name && c[0].dtype==Dtype::F8E4M3 && c[0].shape==w.shape
            && c[1].dtype==Dtype::U8 && c[1].shape==[w.shape[0],w.shape[1]/32],"MXFP8 component layout differs from its declared recipe")?;
        need(op.outputs[0].shape==[w.shape[0]] && op.output_splits==[op.weights[0].split],"MXFP8 output placement differs from its weight rows")?;
        Ok(Mxfp8Projection {shape,origin,total_rows:w.shape[0],topology:p.topology,input:op.inputs[0].clone(),output:op.outputs[0].name.clone(),
            gather:op.gather.contains(&op.outputs[0].name),weight:Component::bind(&c[0],checkpoint,origin,rows,w.shape[1])?,
            scale:Component::bind(&c[1],checkpoint,origin,rows,w.shape[1]/32)?,word:word_of(view),layer:is_layer(view)})
    }
}
impl Mxfp8Projection<'_> {
    pub fn shape(&self)->Shape {self.shape}
    /// The word function of this matrix: `next_params::MX_SCALAR` or `MX_TC`.
    pub fn word(&self)->u32 {self.word}
    /// The same rows on the scalar word. For a check tool that compares a device entry of the scalar word; never a served path.
    pub fn scalar_word(mut self)->Self {self.word=spark_gpu_probe::next_params::MX_SCALAR;self}
    pub fn row_origin(&self)->u64 {self.origin}
    pub fn total_rows(&self)->u64 {self.total_rows}
    pub fn topology(&self)->Topology {self.topology}
    pub fn input_name(&self)->&str {&self.input}
    pub fn output_name(&self)->&str {&self.output}
    pub fn requires_gather(&self)->bool {self.gather}
    pub fn component_names(&self)->[&str;2] {[&self.weight.name,&self.scale.name]}
    pub fn component_offsets(&self)->[u64;2] {[self.weight.offset,self.scale.offset]}
    pub fn component_file_offsets(&self)->[u64;2] {[self.weight.file_offset,self.scale.file_offset]}
    /// Read only complete selected rows, including their matching scale rows.
    pub fn read(&self)->Result<(Vec<u8>,Vec<u8>)> {
        let w=self.weight.read()?;let s=self.scale.read()?;
        need(w.len()==self.shape.weight_bytes() && s.len()==self.shape.scale_bytes(),"MXFP8 selected component size differs")?;
        need(w.iter().enumerate().all(|(i,&v)|engine_model::flash::mxfp8::dequant_value(v,s[i/32]).is_finite()),
            "MXFP8 selected row contains a nonfinite decoded weight")?;
        Ok((w,s))
    }
    /// Retain both original file descriptors through the existing safe arena loader.
    /// Files must remain immutable from header admission through completion.
    #[cfg(target_os="linux")]
    pub fn load<'b,'d>(&self,b:&'b spark_gpu_probe::driver::Backend<'d>)->Result<spark_gpu_probe::driver::ResidentMxfp8Projection<'b,'d>> {
        self.load_shape(b,self.shape)
    }
    #[cfg(target_os="linux")]
    pub(crate) fn load_f32_input<'b,'d>(&self,b:&'b spark_gpu_probe::driver::Backend<'d>)->Result<spark_gpu_probe::driver::ResidentMxfp8Projection<'b,'d>> {
        self.load_shape(b,self.dynamic_shape()?)
    }
    #[cfg(target_os="linux")]
    fn load_shape<'b,'d>(&self,b:&'b spark_gpu_probe::driver::Backend<'d>,shape:Shape)->Result<spark_gpu_probe::driver::ResidentMxfp8Projection<'b,'d>> {
        use spark_gpu_probe::load::{WeightRange,WeightFile};
        b.require_resident_stream()?;self.read()?;
        let ranges=[WeightRange {file_offset:self.weight.file_offset,destination:0,bytes:self.weight.bytes,row_bytes:0,row_stride:0},
            WeightRange {file_offset:self.scale.file_offset,destination:self.weight.bytes,bytes:self.scale.bytes,row_bytes:0,row_stride:0}];
        let arena=b.load_weight_arena(self.shape.arena_bytes()?,&[
            WeightFile {file:self.weight.file.file(),ranges:&ranges[..1]},WeightFile {file:self.scale.file.file(),ranges:&ranges[1..]}])?;
        let matrix=b.resident_mxfp8_projection_mode(&arena,shape,self.word)?;Ok(if self.layer {matrix.counted_as_layer()} else {matrix})
    }
    /// The arena of `load_shape` from the bytes `read` returned (read once, checked there): the E4M3 rows, then their scale rows.
    #[cfg(target_os="linux")]
    pub(crate) fn load_shape_bytes<'b,'d>(&self,b:&'b spark_gpu_probe::driver::Backend<'d>,shape:Shape,weight:&[u8],scale:&[u8])->Result<spark_gpu_probe::driver::ResidentMxfp8Projection<'b,'d>> {
        b.require_resident_stream()?;
        need(weight.len()==self.weight.bytes && scale.len()==self.scale.bytes && weight.len()==shape.weight_bytes() && scale.len()==shape.scale_bytes(),
            "MXFP8 host rows differ from their selected components")?;
        let arena=b.upload_weight_arena(shape.arena_bytes()?,&[(0,weight),(weight.len(),scale)])?;
        let matrix=b.resident_mxfp8_projection_mode(&arena,shape,self.word)?;Ok(if self.layer {matrix.counted_as_layer()} else {matrix})
    }
}
impl<'f> Mxfp8Projection<'f> {
    /// Select complete output rows. Feature order and per-row arithmetic stay fixed.
    pub fn split(&self,topology:Topology)->Result<Self> {
        need(self.origin==0 && self.shape.rows() as u64==self.total_rows,"projection split requires a complete matrix")?;
        topology.validate().map_err(|e|e.to_string())?;
        let r=crate::rows(self.total_rows as usize,topology.rank as usize,topology.parts as usize);
        need(!r.is_empty(),"projection has fewer rows than ranks")?;
        let component=|c:&Component<'f>,stride:usize|->Result<Component<'f>> {
            let offset=r.start.checked_mul(stride).ok_or("projection split offset overflow")?;
            Ok(Component {file:c.file,name:c.name.clone(),offset:c.offset+offset as u64,
                file_offset:c.file_offset+offset as u64,bytes:r.len()*stride})
        };
        Ok(Self {shape:Shape::new(r.len(),self.shape.cols(),self.shape.input())?,origin:r.start as u64,total_rows:self.total_rows,
            topology,input:self.input.clone(),output:self.output.clone(),gather:true,
            weight:component(&self.weight,self.shape.cols())?,scale:component(&self.scale,self.shape.cols()/32)?,word:self.word,layer:self.layer})
    }
    /// Bind one complete matrix by its retained recipe. No producer name tuning.
    pub fn matrix(checkpoint:&'f CheckpointAdmission,name:&str,input:WeightType)->Result<Self> {
        Self::matrix_rows(checkpoint,name,input,Topology {rank:0,parts:1})
    }
    /// Split output rows only. Every rank retains the complete feature axis.
    pub fn matrix_rows(checkpoint:&'f CheckpointAdmission,name:&str,input:WeightType,topology:Topology)->Result<Self> {
        let view=checkpoint.headers().weights().get(name).ok_or("missing MXFP8 matrix")?;let w=view.logical();
        need(w.storage==WeightStorage::Mxfp8 && w.shape.len()==2,"matrix requires an MXFP8 recipe")?;
        let ownership=engine_model::plan::tensor_split::Ownership::equal(&w.shape,0,1,topology.parts).map_err(|e|e.to_string())?;
        let interval=ownership.owner(topology.rank).map_err(|e|e.to_string())?.interval.clone();let rows=interval.end-interval.start;
        let shape=Shape::new(rows.try_into().map_err(|_|"MXFP8 rows exceed address space")?,w.shape[1].try_into().map_err(|_|"MXFP8 columns exceed address space")?,input)?;
        let c=view.components();need(c.len()==2 && c[0].dtype==Dtype::F8E4M3 && c[0].shape==w.shape
            && c[1].dtype==Dtype::U8 && c[1].shape==[w.shape[0],w.shape[1]/32],"MXFP8 matrix components differ")?;
        Ok(Self {shape,origin:interval.start,total_rows:w.shape[0],topology,input:String::new(),output:String::new(),gather:rows!=w.shape[0],
            weight:Component::bind(&c[0],checkpoint,interval.start,rows,w.shape[1])?,scale:Component::bind(&c[1],checkpoint,interval.start,rows,w.shape[1]/32)?,word:word_of(view),layer:is_layer(view)})
    }
}

impl<'f> Mxfp8Projection<'f> {
    /// Bind a complete output head from the admitted storage recipe and headers.
    pub fn bind_head(checkpoint:&'f CheckpointAdmission,name:&str)->Result<Self> {
        let view=checkpoint.headers().weights().get(name).ok_or("head weight is absent")?;let w=view.logical();
        need(view.binding().operation==engine_model::plan::glm_index::TensorRole::Head && w.storage==WeightStorage::Mxfp8 && w.shape==[checkpoint.description().vocab,checkpoint.description().hidden],"head matrix differs from text geometry")?;
        let shape=Shape::new(w.shape[0].try_into().map_err(|_|"head rows exceed address space")?,
            w.shape[1].try_into().map_err(|_|"head columns exceed address space")?,WeightType::F32)?;
        let c=view.components();
        need(c.len()==2 && c[0].name==w.name && c[0].dtype==Dtype::F8E4M3 && c[0].shape==w.shape
            && c[1].dtype==Dtype::U8 && c[1].shape==[w.shape[0],w.shape[1]/32],"head MXFP8 component layout differs")?;
        Ok(Self {shape,origin:0,total_rows:w.shape[0],topology:Topology {rank:0,parts:1},input:"final_norm".into(),output:"logits".into(),gather:false,
            weight:Component::bind(&c[0],checkpoint,0,w.shape[0],w.shape[1])?,scale:Component::bind(&c[1],checkpoint,0,w.shape[0],w.shape[1]/32)?,
            word:spark_gpu_probe::next_params::MX_SCALAR,layer:false})
    }
}
