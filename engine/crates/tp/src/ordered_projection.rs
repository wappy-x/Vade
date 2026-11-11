//! Bind a logical FP32 projection to retained plain FP32 checkpoint rows.
//! Effective checkpoint runtime loads and complete layer execution stay refused.
use engine_model::{plan::{layer::{Kind,LayerPlan,Split,profile::{Profile,UnsupportedProfile}},storage::WeightStorage,
    tensor_split::{Ownership,ProjectionRows}},
    safetensors::{Dtype,TensorFile}};
use spark_gpu_probe::ordered_projection::Shape;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error { Profile(UnsupportedProfile), Operation, Storage, Header, Bounds, Plan(String), Device(String) }
impl fmt::Display for Error {
    fn fmt(&self,f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(e)=>e.fmt(f),Self::Operation=>f.write_str("step is not a single ordinary projection"),
            Self::Storage=>f.write_str("logical projection requires plain FP32 storage"),
            Self::Header=>f.write_str("projection header differs from the logical weight"),
            Self::Bounds=>f.write_str("projection row or byte range is outside the tensor"),
            Self::Plan(e)|Self::Device(e)=>f.write_str(e),
        }
    }
}
impl std::error::Error for Error {}

/// The borrow retains the validated file descriptor. It does not prove payload identity.
pub struct PlainF32Projection<'f> { file: &'f TensorFile, name: String, shape: Shape, origin: u64, byte_offset: u64, file_offset: u64 }
impl<'f> PlainF32Projection<'f> {
    pub fn bind(plan: &LayerPlan,step: usize,file: &'f TensorFile,profile: Profile) -> Result<Self,Error> {
        profile.require_scalar().map_err(Error::Profile)?;
        plan.validate().map_err(|e| Error::Plan(e.to_string()))?;
        let op=plan.steps.get(step).ok_or(Error::Operation)?;
        if op.kind!=Kind::Project || op.weights.len()!=1 {return Err(Error::Operation);}
        let w=&op.weights[0];
        if w.storage!=WeightStorage::Plain {return Err(Error::Storage);}
        let t=file.header().tensors.get(&w.name).ok_or(Error::Header)?;
        if t.dtype!=Dtype::F32 {return Err(Error::Storage);}
        if w.shape.len()!=2 || t.shape!=w.shape {return Err(Error::Header);}
        let (origin,rows)=match w.split {
            Split::Replicated=>(0,w.shape[0]),Split::Axis {axis,unit}=> {
                let ownership=Ownership::equal(&w.shape,axis,unit,plan.topology.parts).map_err(|e| Error::Plan(e.to_string()))?;
                let range=ProjectionRows::new(&ownership,plan.topology.rank,w.storage).map_err(|e| Error::Plan(e.to_string()))?.rows();
                (range.start,range.end-range.start)
            },
        };
        let shape=Shape::new(usize::try_from(rows).map_err(|_| Error::Bounds)?,
            usize::try_from(w.shape[1]).map_err(|_| Error::Bounds)?).map_err(|_| Error::Bounds)?;
        let byte_offset=origin.checked_mul(w.shape[1]).and_then(|n| n.checked_mul(4)).ok_or(Error::Bounds)?;
        let end=byte_offset.checked_add(shape.weight_bytes() as u64).ok_or(Error::Bounds)?;
        if end>t.byte_len() {return Err(Error::Bounds);}
        let file_offset=file.header().data_start.checked_add(t.start).and_then(|n| n.checked_add(byte_offset)).ok_or(Error::Bounds)?;
        if file_offset.checked_add(shape.weight_bytes() as u64).is_none_or(|n| n>file.header().file_bytes) {return Err(Error::Bounds);}
        Ok(Self {file,name:w.name.clone(),shape,origin,byte_offset,file_offset})
    }
    pub fn shape(&self) -> Shape {self.shape}
    pub fn row_origin(&self) -> u64 {self.origin}
    pub fn file_offset(&self) -> u64 {self.file_offset}
    /// Host inspection uses the same retained descriptor and selected byte range.
    pub fn read(&self) -> Result<Vec<f32>,Error> {
        let mut bytes=vec![0;self.shape.weight_bytes()];
        self.file.read_range(&self.name,self.byte_offset,&mut bytes).map_err(|e| Error::Plan(e.to_string()))?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }
    /// Load original FP32 bytes through the existing arena loader and matrix type.
    #[cfg(target_os="linux")]
    pub fn load<'a>(&self,backend: &spark_gpu_probe::driver::Backend<'a>) -> Result<spark_gpu_probe::driver::Matrix<'a>,Error> {
        use spark_gpu_probe::load::{WeightRange,WeightFile,WeightDtype};
        let ranges=[WeightRange {file_offset:self.file_offset,destination:0,bytes:self.shape.weight_bytes(),row_bytes:0,row_stride:0}];
        let arena=backend.load_weight_arena(self.shape.weight_bytes(),&[WeightFile {file:self.file.file(),ranges:&ranges}])
            .map_err(Error::Device)?;
        backend.weight_matrix(&arena,0,self.shape.rows(),self.shape.cols(),WeightDtype::F32).map_err(Error::Device)
    }
}
#[cfg(test)]
#[path="ordered_projection/tests.rs"] mod tests;
