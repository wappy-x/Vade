//! Header admission and upload byte views for deferred text operations.
//! This module does not grant execution or select kernels.
use super::{TextDescription, description::*, glm_index::*, layer::{LayerPlan, Topology}, storage::*};
use crate::{Result, invalid, safetensors::{self, Dtype, Tensor, TensorFile}, shapes::*};
use std::{collections::{BTreeMap, BTreeSet}, path::Path};

fn need(ok: bool, why: impl Into<String>) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn mul(a: u64,b: u64) -> Result<u64> { a.checked_mul(b).ok_or_else(|| invalid("storage extent overflow")) }
fn add(a: u64,b: u64) -> Result<u64> { a.checked_add(b).ok_or_else(|| invalid("storage extent overflow")) }
fn extent(shape: &[u64],dtype: Dtype) -> Result<u64> { shape.iter().try_fold(dtype.bytes(),|n,&d| mul(n,d)) }
fn limits() -> safetensors::Limits { safetensors::Limits { header_bytes: HEADER_BYTES,
    tensors: crate::flash::mixed::MAX_MIXED_TENSORS,rank:MAX_TENSOR_RANK,shards:MAX_SHARDS } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalWeight { pub name: String, pub shape: Vec<u64>, pub storage: WeightStorage }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentView { pub name: String, pub dtype: Dtype, pub shape: Vec<u64>, pub start: u64, pub bytes: u64 }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightView { logical: LogicalWeight, binding: TensorBinding, components: Vec<ComponentView> }
#[derive(Debug)]
pub struct HeaderAdmission { weights: BTreeMap<String, WeightView>, pub stored_bytes: u64 }
/// Retain the descriptors that supplied the admitted headers. Files must stay immutable.
pub struct CheckpointAdmission { headers: HeaderAdmission, files: BTreeMap<String,TensorFile>, locations: BTreeMap<String,String> }

impl TextDescription {
    /// Required text weights include stored draft data. This does not define a draft forward pass.
    pub fn required_weights(&self) -> Result<BTreeMap<String,LogicalWeight>> {
        need(!self.layers.is_empty() && self.layers.len()<=MAX_LAYERS as usize
            && self.declared_draft_layers<=MAX_LAYERS,"layer count exceeds storage limits")?;
        let mut weights=BTreeMap::new();
        let mut insert=|mut name: String,shape: Vec<u64>,storage: WeightStorage| -> Result<()> {
            if name.ends_with(".indexer.index_kpool_compress_gate.weight") { name.truncate(name.len()-7); }
            need(self.bind_tensor(&name)?.storage==storage,format!("storage recipe differs from operation storage for {name}"))?;
            extent(&shape,Dtype::F32)?;
            need(shape.iter().all(|&d| d>0),"empty logical weight")?;
            need(weights.len()<limits().tensors,"logical tensor limit exceeded")?;
            if let Some(old)=weights.insert(name.clone(),LogicalWeight { name,shape:shape.clone(),storage }) {
                need(old.shape==shape && old.storage==storage,"conflicting logical weight")?;
            } Ok(())
        };
        for (name,shape) in [("model.language_model.embed_tokens.weight",vec![self.vocab,self.hidden]),
            ("lm_head.weight",vec![self.vocab,self.hidden]),("model.language_model.norm.weight",vec![self.hidden])] {
            let module=name.strip_suffix(".weight").unwrap(); insert(name.into(),shape,self.storage.storage(module))?;
        }
        for layer in 0..self.layers.len() {
            let p=LayerPlan::new(self,layer,1,Topology { parts:1,rank:0 })?;
            for w in p.steps.into_iter().flat_map(|s|s.weights) { insert(w.name,w.shape,w.storage)?; }
        }
        // The source schema gives appended layers the common latent and expert geometry.
        // Refuse ambiguity instead of selecting a decoder layer by position.
        if self.declared_draft_layers>0 {
            let mut latent=None; let mut experts=None;
            for l in &self.layers {
                if let TokenMixer::LatentAttention(a)=l.mixer {
                    need(latent.is_none_or(|v|v==a),"ambiguous stored draft attention geometry")?; latent=Some(a);
                }
                if let Ffn::Experts(e)=l.ffn {
                    need(experts.is_none_or(|v|v==e),"ambiguous stored draft expert geometry")?; experts=Some(e);
                }
            }
            let decoder=DecoderDescription { mixer:TokenMixer::LatentAttention(latent.ok_or_else(||invalid("missing draft attention geometry"))?),
                ffn:Ffn::Experts(experts.ok_or_else(||invalid("missing draft expert geometry"))?) };
            let mut d=self.clone();
            let drafts=usize::try_from(self.declared_draft_layers).map_err(|_|invalid("draft count overflow"))?;
            need(drafts<=MAX_LAYERS as usize,"draft count exceeds limit")?;
            for layer in self.layers.len()..self.layers.len().checked_add(drafts).ok_or_else(||invalid("layer count overflow"))? {
                d.layers.push(decoder);
                for w in LayerPlan::new(&d,layer,1,Topology { parts:1,rank:0 })?.steps.into_iter().flat_map(|s|s.weights) {
                    if !matches!(d.bind_tensor(&w.name).map(|b|b.operation),Ok(TensorRole::Mhc { .. })) {
                        insert(w.name,w.shape,w.storage)?;
                    }
                }
                let prefix=format!("model.language_model.layers.{layer}");
                for (suffix,shape) in [("eh_proj.weight",vec![self.hidden,mul(2,self.hidden)?]),("enorm.weight",vec![self.hidden]),
                    ("hnorm.weight",vec![self.hidden]),("shared_head.norm.weight",vec![self.hidden])] {
                    let name=format!("{prefix}.{suffix}"); let module=name.strip_suffix(".weight").unwrap();
                    insert(name.clone(),shape,self.storage.storage(module))?;
                }
            }
        }
        need(weights.len()<=crate::flash::mixed::MAX_MIXED_TENSORS,"logical tensor limit exceeded")?;
        Ok(weights)
    }
    /// Check all text names, physical axes, component dtypes and byte extents.
    /// Callers with files must use `open_headers` to check shard and index coverage too.
    pub fn admit_headers(&self,tensors: &BTreeMap<String,Tensor>) -> Result<HeaderAdmission> {
        need(tensors.len()<=limits().tensors,"header tensor limit exceeded")?;
        let mut seen=BTreeSet::new(); let mut weights=BTreeMap::new(); let mut stored_bytes=0;
        for (name,t) in tensors {
            let bytes=extent(&t.shape,t.dtype)?;
            need(t.shape.len()<=MAX_TENSOR_RANK && t.end.checked_sub(t.start)==Some(bytes),format!("invalid byte extent for {name}"))?;
            stored_bytes=add(stored_bytes,bytes)?;
        }
        for (name,logical) in self.required_weights()? {
            let view=self.admit_weight(logical,tensors)?;
            seen.extend(view.components.iter().map(|c|c.name.clone()));
            weights.insert(name,view);
        }
        for name in tensors.keys() {
            let binding=self.bind_tensor(name)?;
            need(binding.operation==TensorRole::Vision || seen.contains(name),format!("unexpected text tensor {name}"))?;
        }
        Ok(HeaderAdmission {weights,stored_bytes})
    }
    fn admit_weight(&self,logical: LogicalWeight,tensors: &BTreeMap<String,Tensor>) -> Result<WeightView> {
        let name=logical.name.clone();
        let binding=self.bind_tensor(&name)?;
        let mut components=Vec::new();
        let mut check=|component: String,dtype: &[Dtype],shape: Option<Vec<u64>>| -> Result<()> {
            let t=tensors.get(&component).ok_or_else(||invalid(format!("missing tensor {component}")))?;
            need(dtype.contains(&t.dtype) && shape.as_ref().is_none_or(|s| *s==t.shape),format!("wrong dtype or shape for {component}"))?;
            if shape.is_none() { need(t.shape.len()==1 && t.shape[0]<=mul(logical.shape[0],logical.shape[1]/16)?
                && t.end-t.start<=u32::MAX as u64,"invalid CSF exception extent")?; }
            components.push(ComponentView { name:component,dtype:t.dtype,shape:t.shape.clone(),start:t.start,bytes:t.end-t.start }); Ok(())
        };
        let mut physical=logical.shape.clone();
        if matches!(binding.operation,TensorRole::Kda(KdaTensor::QueryConv | KdaTensor::KeyConv | KdaTensor::ValueConv)) {
            physical.insert(1,1);
        }
        let module=name.strip_suffix(".weight").unwrap_or(&name);
        match logical.storage {
            WeightStorage::Plain => {
                let types: &[Dtype]=match binding.operation {
                    TensorRole::Mhc { parameter:MhcTensor::Base | MhcTensor::Scale,.. } | TensorRole::Kda(KdaTensor::Decay | KdaTensor::TimeBias)
                        | TensorRole::RouterBias => &[Dtype::F32], _ => &[Dtype::Bf16,Dtype::F32],
                };
                check(name.clone(),types,Some(physical))?;
            },
            storage => {
                need(logical.shape.len()==2,"quantized storage requires a matrix")?;
                let (r,c)=(logical.shape[0],logical.shape[1]);
                match storage {
                    WeightStorage::Mxfp8 => {
                        need(c%32==0,"MXFP8 columns must cover complete scale blocks")?;
                        check(name.clone(),&[Dtype::F8E4M3],Some(vec![r,c]))?;
                        check(format!("{module}.weight_scale"),&[Dtype::U8],Some(vec![r,c/32]))?;
                    },
                    WeightStorage::Fp8Block {block:[br,bc]} => {
                        need(br>0 && bc>0,"empty FP8 scale block")?;
                        check(name.clone(),&[Dtype::F8E4M3],Some(vec![r,c]))?;
                        // ModelOpt retains both broadcast axes in the stored scale plane.
                        check(format!("{module}.weight_scale"),&[Dtype::F32],Some(vec![r.div_ceil(br),1,c.div_ceil(bc),1]))?;
                    },
                    WeightStorage::Nvfp4 {encoding,activation_scale} => {
                        need(c%16==0,"NVFP4 columns must cover complete scale blocks")?;
                        check(name.clone(),&[Dtype::U8],Some(vec![r,c/2]))?;
                        match encoding {
                            ScaleEncoding::Direct => check(format!("{module}.weight_scale"),&[Dtype::F8E4M3],Some(vec![r,c/16]))?,
                            ScaleEncoding::Csf32 => {
                                need(r%32==0,"CSF32 rows must cover complete blocks")?;
                                let cols=c/16; let blocks=mul(r/32,cols.div_ceil(2))?;
                                let pages=mul(r.div_ceil(128),cols.div_ceil(4))?.div_ceil(32);
                                for (suffix,dtype,shape) in [("bases",Dtype::U8,Some(vec![r])),("codes",Dtype::U8,Some(vec![mul(r,cols)?/2])),
                                    ("page_bases",Dtype::U32,Some(vec![pages])),("offsets",Dtype::U16,Some(vec![add(blocks,pages)?])),("exceptions",Dtype::U16,None)] {
                                    check(format!("{module}.weight_scale.csf32_{suffix}"),&[dtype],shape)?;
                                }
                            },
                            ScaleEncoding::Csf => return Err(invalid("legacy CSF header admission is not supported by this description")),
                        }
                        check(format!("{module}.weight_scale_2"),&[Dtype::F32],Some(vec![]))?;
                        if activation_scale { check(format!("{module}.input_scale"),&[Dtype::F32],Some(vec![]))?; }
                    },
                    WeightStorage::Plain => unreachable!(),
                }
            },
        }
        Ok(WeightView {logical,binding,components})
    }
    /// Header-only file admission. No tensor payload or device allocation is read here.
    pub fn open_headers(&self,root: &Path,index_bytes: &[u8]) -> Result<CheckpointAdmission> {
        let mapped=self.map_index(index_bytes)?; let root=root.canonicalize()?;
        let mut files=BTreeMap::new(); let mut tensors=BTreeMap::new();
        for name in mapped.index.weight_map.values().collect::<BTreeSet<_>>() {
            let path=root.join(name).canonicalize()?;
            need(path.starts_with(&root),"shard symlink escapes checkpoint directory")?;
            let file=TensorFile::open(&path,limits())?;
            for (tensor,t) in &file.header().tensors {
                need(mapped.index.weight_map.get(tensor)==Some(name) && !tensors.contains_key(tensor),"shard tensor differs from index")?;
                tensors.insert(tensor.clone(),t.clone());
            }
            files.insert(name.clone(),file);
        }
        need(tensors.len()==mapped.index.weight_map.len(),"index tensor is absent from shard")?;
        let headers=self.admit_headers(&tensors)?;
        need(!mapped.index.size_declared || mapped.index.total_size==headers.stored_bytes,"index total_size differs from stored bytes")?;
        Ok(CheckpointAdmission {headers,files,locations:mapped.index.weight_map})
    }
}

/// Byte planes ready for upload. CSF32 scales expand on the host to existing NVFP4 bytes.
#[derive(Debug)]
pub struct UploadWeight { pub shape: Vec<u64>, pub dtype: Dtype, pub weight: Vec<u8>, pub scales: Vec<u8>,
    pub global_scale: Option<f32>, pub input_scale: Option<f32>, pub storage: WeightStorage }
impl HeaderAdmission {
    pub fn weights(&self) -> &BTreeMap<String,WeightView> { &self.weights }
}
impl WeightView {
    pub fn logical(&self) -> &LogicalWeight { &self.logical }
    pub fn binding(&self) -> TensorBinding { self.binding }
    pub fn components(&self) -> &[ComponentView] { &self.components }
    pub fn upload(&self,mut read: impl FnMut(&ComponentView)->Result<Vec<u8>>) -> Result<UploadWeight> {
        let mut bytes=BTreeMap::new();
        for c in &self.components { let b=read(c)?; need(b.len() as u64==c.bytes,"component payload byte extent differs from header")?; bytes.insert(c.name.clone(),b); }
        let name=&self.logical.name; let module=name.strip_suffix(".weight").unwrap_or(name);
        let weight=bytes.remove(name).ok_or_else(||invalid("missing admitted weight bytes"))?;
        let scalar=|suffix: &str| -> Result<Option<f32>> {
            bytes.get(&format!("{module}.{suffix}")).map(|b| {
                let v=f32::from_le_bytes(b.as_slice().try_into().map_err(|_|invalid("invalid scale scalar bytes"))?);
                need(v.is_finite() && v>0.0,"scale scalar must be positive and finite")?; Ok(v)
            }).transpose()
        };
        let global_scale=scalar("weight_scale_2")?; let input_scale=scalar("input_scale")?;
        let scales=if matches!(self.logical.storage,WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,..}) {
            let get=|suffix: &str| -> Result<&[u8]> { bytes.get(&format!("{module}.weight_scale.csf32_{suffix}")).map(Vec::as_slice).ok_or_else(||invalid("missing CSF32 bytes")) };
            crate::csf::Csf32 { rows:usize::try_from(self.logical.shape[0]).map_err(|_|invalid("row count overflow"))?,
                columns:usize::try_from(self.logical.shape[1]/16).map_err(|_|invalid("column count overflow"))?,quantization:crate::csf::Quantization::Nvfp4,
                bases:get("bases")?,codes:get("codes")?,page_bases:get("page_bases")?,offsets:get("offsets")?,exceptions:get("exceptions")? }.expand()?
        } else { bytes.remove(&format!("{module}.weight_scale")).unwrap_or_default() };
        if matches!(self.logical.storage,WeightStorage::Fp8Block {..}) {
            need(scales.chunks_exact(4).all(|b| { let v=f32::from_le_bytes(b.try_into().unwrap()); v.is_finite() && v>0.0 }),"FP8 block scale must be positive and finite")?;
        }
        if self.logical.storage==WeightStorage::Mxfp8 { need(!scales.contains(&255),"invalid MXFP8 scale byte")?; }
        if matches!(self.logical.storage,WeightStorage::Nvfp4 {..}) {
            need(scales.iter().all(|&b| {let v=crate::flash::nvfp4::e4m3(b); v.is_finite() && v>=0.0}),"invalid NVFP4 scale byte")?;
        }
        Ok(UploadWeight {shape:self.logical.shape.clone(),dtype:self.components[0].dtype,weight,scales,global_scale,input_scale,storage:self.logical.storage})
    }
}

/// A row-strided matrix view. Scales retain their complete original plane.
/// `origin` is global, including the position within the first scale block.
#[derive(Debug)]
pub struct MatrixByteView<'a> {
    pub weight: &'a [u8], pub row_stride: u64, pub shape: [u64;2], pub origin: [u64;2],
    pub scales: &'a [u8], pub scale_grid: [u64;2], pub scale_block: [u64;2], pub scale_origin: [u64;2],
    pub dtype: Dtype, pub global_scale: Option<f32>, pub input_scale: Option<f32>,
}
impl UploadWeight {
    pub fn matrix_view(&self,origin: [u64;2],shape: [u64;2]) -> Result<MatrixByteView<'_>> {
        need(self.shape.len()==2,"matrix byte view requires matrix storage")?;
        for axis in 0..2 { need(shape[axis]>0 && add(origin[axis],shape[axis])?<=self.shape[axis],"matrix view exceeds weight")?; }
        let block=match self.storage { WeightStorage::Plain => [1,1],WeightStorage::Mxfp8 => [1,32],
            WeightStorage::Fp8Block {block} => block,WeightStorage::Nvfp4 {..} => [1,16] };
        need(block.iter().all(|&v|v>0),"empty matrix scale block")?;
        let packed=matches!(self.storage,WeightStorage::Nvfp4 {..});
        let divisor=if packed {2} else {1};
        need(!packed || self.shape[1]%2==0 && origin[1]%2==0 && shape[1]%2==0,"packed view must cover complete bytes")?;
        let stride=mul(self.shape[1]/divisor,self.dtype.bytes())?;
        need(self.weight.len() as u64==mul(self.shape[0],stride)?,"weight byte plane differs from matrix")?;
        let start=add(mul(origin[0],stride)?,mul(origin[1]/divisor,self.dtype.bytes())?)?;
        let end=add(start,add(mul(shape[0]-1,stride)?,mul(shape[1]/divisor,self.dtype.bytes())?)?)?;
        let start=usize::try_from(start).map_err(|_|invalid("view offset exceeds address space"))?;
        let end=usize::try_from(end).map_err(|_|invalid("view offset exceeds address space"))?;
        let grid=[self.shape[0].div_ceil(block[0]),self.shape[1].div_ceil(block[1])];
        let scale_bytes=match self.storage { WeightStorage::Plain => 0,WeightStorage::Fp8Block {..} => mul(mul(grid[0],grid[1])?,4)?,
            _ => mul(grid[0],grid[1])? };
        need(self.scales.len() as u64==scale_bytes,"scale byte plane differs from matrix")?;
        Ok(MatrixByteView {weight:&self.weight[start..end],row_stride:stride,shape,origin,scales:&self.scales,
            scale_grid:grid,scale_block:block,scale_origin:[origin[0]/block[0],origin[1]/block[1]],dtype:self.dtype,
            global_scale:self.global_scale,input_scale:self.input_scale})
    }
}
impl CheckpointAdmission {
    pub fn headers(&self) -> &HeaderAdmission { &self.headers }
    pub fn upload(&self,name: &str) -> Result<UploadWeight> {
        let view=self.headers.weights.get(name).ok_or_else(||invalid("weight is not admitted"))?;
        view.upload(|c| {
            let mut b=Vec::new(); let n=usize::try_from(c.bytes).map_err(|_|invalid("component exceeds address space"))?;
            b.try_reserve_exact(n).map_err(|_|invalid("component allocation failed"))?; b.resize(n,0);
            let file=self.locations.get(&c.name).and_then(|s|self.files.get(s)).ok_or_else(||invalid("missing retained tensor file"))?;
            file.read_range(&c.name,0,&mut b)?; Ok(b)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_geometry_rejects_partial_blocks_and_nonmatrix_quantization() {
        let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../tests/fixtures/glm/config.json")).unwrap() else {panic!()};
        for (shape,storage) in [(vec![32,63],WeightStorage::Mxfp8),
            (vec![32,63],WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:false}),
            (vec![31,64],WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:false}),
            (vec![32,64],WeightStorage::Fp8Block {block:[0,16]}),
            (vec![32,64],WeightStorage::Fp8Block {block:[16,0]}),
            (vec![32],WeightStorage::Mxfp8),
            (vec![32,64],WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf,activation_scale:false})] {
            let packed=matches!(storage,WeightStorage::Nvfp4 {..});
            let physical=if packed {vec![shape[0],shape[1]/2]} else {shape.clone()};
            let dtype=if packed {Dtype::U8} else {Dtype::F8E4M3};
            let bytes=extent(&physical,dtype).unwrap();
            let t=BTreeMap::from([("lm_head.weight".into(),Tensor {dtype,shape:physical,start:0,end:bytes})]);
            let w=LogicalWeight {name:"lm_head.weight".into(),shape,storage};
            assert!(d.admit_weight(w,&t).is_err());
        }
    }
    #[test]
    fn recorded_public_headers_match_operation_and_storage_contracts() {
        let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("../../tests/fixtures/glm/config.json")).unwrap() else {panic!()};
        let j=crate::json::parse(include_bytes!("../../tests/fixtures/glm/public-header-samples.json"),crate::json::Limits::default()).unwrap();
        let mut samples=BTreeMap::new();
        for (name,v) in j.object().unwrap() {
            let o=v.object().unwrap();let a=o["data_offsets"].array().unwrap();
            samples.insert(name.clone(),Tensor {dtype:Dtype::parse(o["dtype"].string().unwrap()).unwrap(),
                shape:o["shape"].array().unwrap().iter().map(|v|v.u64().unwrap()).collect(),start:a[0].u64().unwrap(),end:a[1].u64().unwrap()});
        }
        let mut seen=BTreeSet::new();
        for (_,w) in d.required_weights().unwrap() {
            if samples.contains_key(&w.name) {
                let view=d.admit_weight(w,&samples).unwrap();seen.extend(view.components.iter().map(|c|c.name.clone()));
            }
        }
        assert_eq!(seen,samples.keys().cloned().collect());
    }
}
