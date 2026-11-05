//! Dynamic MXFP8 activation binding. Complete layer admission stays separate.
use crate::{Result,mxfp8_projection::Mxfp8Projection};
use spark_gpu_probe::{dynamic_mxfp8::Activation,mxfp8_projection::Shape,norm_boundary::WeightType};

impl Mxfp8Projection<'_> {
    /// The recipe rounds source activations to BF16 before dynamic quantization.
    /// Decoded MXFP8 values reach the fixed FP32 projection without another cast.
    pub fn dynamic_reference(&self,input:&[f32])->Result<(Activation,Vec<f32>,Vec<f32>)> {
        self.shape().check_input(input)?;
        let activation=Activation::from_f32(input)?;let (weight,scale)=self.read()?;
        let (accumulator,output)=self.dynamic_shape()?.reference(&weight,&scale,activation.decoded())?;
        Ok((activation,accumulator,output))
    }
    pub fn dynamic_shape(&self)->Result<Shape> {Shape::new(self.shape().rows(),self.shape().cols(),WeightType::F32)}
    /// Keep the original row descriptors. Only the activation contract changes.
    #[cfg(target_os="linux")]
    pub fn load_dynamic<'b,'d>(&self,backend:&'b spark_gpu_probe::driver::Backend<'d>)->Result<ResidentDynamicProjection<'b,'d>> {
        let projection=self.load_f32_input(backend)?;
        Ok(ResidentDynamicProjection {backend,source:self.shape(),projection})
    }
}

#[cfg(target_os="linux")]
pub struct ResidentDynamicProjection<'b,'d> {
    backend:&'b spark_gpu_probe::driver::Backend<'d>,source:Shape,
    projection:spark_gpu_probe::driver::ResidentMxfp8Projection<'b,'d>,
}
#[cfg(target_os="linux")]
pub struct DynamicProjectionPass<'d> {
    activation:Activation,projection:spark_gpu_probe::driver::Mxfp8ProjectionPass<'d>,
}
#[cfg(target_os="linux")]
impl<'d> ResidentDynamicProjection<'_,'d> {
    /// Quantize the complete BF16-source groups on the device. The projection
    /// keeps its original feature tree. No activation crosses the host boundary.
    pub fn project_vector(&self,input:&spark_gpu_probe::driver::DeviceVector<'_>)->Result<spark_gpu_probe::driver::Mxfp8ProjectionPass<'d>> {
        use spark_gpu_probe::{causal_ops::{Shape,Op},driver::NormOutput};
        let b=self.backend;b.require_resident_stream()?;
        if input.len()!=self.source.cols() || input.is_segmented() {return Err("dynamic MXFP8 input extent differs".into());}
        let s=Shape {op:Op::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:0,cols:input.len().try_into().map_err(|_|"MXFP8 columns exceed ABI")?,epsilon:0.0,bound:0.0};
        let decoded=b.causal_operation(s,[Some(input),None,None,None],None)?;
        self.projection.apply(&NormOutput::F32(decoded))
    }
    /// The bounded binding stages activations through the host quantizer.
    /// Readback and upload complete on the same backend. This is no speed claim.
    pub fn apply(&self,input:&spark_gpu_probe::driver::NormOutput<'_>)->Result<DynamicProjectionPass<'d>> {
        use spark_gpu_probe::driver::NormOutput;
        let b=self.backend;b.require_resident_stream()?;
        let dtype=match input {NormOutput::Bf16(_)=>WeightType::Bf16,NormOutput::F32(_)=>WeightType::F32};
        if dtype!=self.source.input() || input.vector().len()!=self.source.cols() || input.vector().is_segmented() {
            return Err("dynamic MXFP8 input differs from its complete producer row".into());
        }
        self.apply_vector(input.vector())
    }
    pub fn apply_vector(&self,input:&spark_gpu_probe::driver::DeviceVector<'_>)->Result<DynamicProjectionPass<'d>> {
        let b=self.backend;b.require_resident_stream()?;
        if input.len()!=self.source.cols() || input.is_segmented() {return Err("dynamic MXFP8 input extent differs".into());}
        let values=b.read_vector_values(input)?;self.source.check_input(&values)?;
        let activation=Activation::from_f32(&values)?;
        let uploaded=spark_gpu_probe::driver::NormOutput::F32(b.upload_vector(activation.decoded())?);b.synchronize()?;
        let projection=self.projection.apply(&uploaded)?;
        Ok(DynamicProjectionPass {activation,projection})
    }
}
#[cfg(target_os="linux")]
impl<'d> DynamicProjectionPass<'d> {
    pub fn activation(&self)->&Activation {&self.activation}
    pub fn projection(&self)->&spark_gpu_probe::driver::Mxfp8ProjectionPass<'d> {&self.projection}
}
