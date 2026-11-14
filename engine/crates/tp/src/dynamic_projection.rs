//! Dynamic MXFP8 activation binding. Complete layer admission stays separate.
use crate::{Result,mxfp8_projection::Mxfp8Projection};
use spark_gpu_probe::{dynamic_mxfp8::Activation,mxfp8_projection::Shape,norm_boundary::WeightType};

impl Mxfp8Projection<'_> {
    /// The recipe rounds source activations to BF16 before dynamic quantization.
    /// Decoded MXFP8 values reach the fixed FP32 projection without another cast.
    pub fn dynamic_reference(&self,input:&[f32])->Result<(Activation,Vec<f32>,Vec<f32>)> {
        self.reference_with_zero_sign(input,false)
    }
    pub fn dynamic_reference_positive_zero(&self,input:&[f32])->Result<(Activation,Vec<f32>,Vec<f32>)> {
        self.reference_with_zero_sign(input,true)
    }
    fn reference_with_zero_sign(&self,input:&[f32],positive_zero:bool)->Result<(Activation,Vec<f32>,Vec<f32>)> {
        self.shape().check_input(input)?;
        let activation=if positive_zero {Activation::from_f32_positive_zero(input)?} else {Activation::from_f32(input)?};let (weight,scale)=self.read()?;
        let (accumulator,output)=self.dynamic_shape()?.reference(&weight,&scale,activation.decoded())?;
        Ok((activation,accumulator,output))
    }
    pub fn dynamic_shape(&self)->Result<Shape> {Shape::new(self.shape().rows(),self.shape().cols(),WeightType::F32)}
    /// Keep the original row descriptors. Only the activation contract changes.
    #[cfg(target_os="linux")]
    pub fn load_dynamic<'b,'d>(&self,backend:&'b spark_gpu_probe::driver::Backend<'d>)->Result<ResidentDynamicProjection<'b,'d>> {
        self.load_dynamic_with_zero_sign(backend,false)
    }
    #[cfg(target_os="linux")]
    pub fn load_dynamic_positive_zero<'b,'d>(&self,backend:&'b spark_gpu_probe::driver::Backend<'d>)->Result<ResidentDynamicProjection<'b,'d>> {
        self.load_dynamic_with_zero_sign(backend,true)
    }
    /// `load_dynamic_positive_zero` from the rows `read` returned, with no second read of the file.
    #[cfg(target_os="linux")]
    pub fn load_dynamic_positive_zero_bytes<'b,'d>(&self,backend:&'b spark_gpu_probe::driver::Backend<'d>,weight:&[u8],scale:&[u8])->Result<ResidentDynamicProjection<'b,'d>> {
        let projection=self.load_shape_bytes(backend,self.dynamic_shape()?,weight,scale)?;
        Ok(ResidentDynamicProjection {backend,source:self.shape(),projection,positive_zero:true})
    }
    #[cfg(target_os="linux")]
    fn load_dynamic_with_zero_sign<'b,'d>(&self,backend:&'b spark_gpu_probe::driver::Backend<'d>,positive_zero:bool)->Result<ResidentDynamicProjection<'b,'d>> {
        let projection=self.load_f32_input(backend)?;
        Ok(ResidentDynamicProjection {backend,source:self.shape(),projection,positive_zero})
    }
}

#[cfg(target_os="linux")]
pub struct ResidentDynamicProjection<'b,'d> {
    backend:&'b spark_gpu_probe::driver::Backend<'d>,source:Shape,
    projection:spark_gpu_probe::driver::ResidentMxfp8Projection<'b,'d>,
    positive_zero:bool,
}
#[cfg(target_os="linux")]
pub struct DynamicProjectionPass<'d> {
    activation:Activation,projection:spark_gpu_probe::driver::Mxfp8ProjectionPass<'d>,
}
#[cfg(target_os="linux")]
impl<'d> ResidentDynamicProjection<'_,'d> {
    pub fn resident(&self)->&spark_gpu_probe::driver::ResidentMxfp8Projection<'_,'d> {&self.projection}
    /// Quantize the complete BF16-source groups on the device. The projection
    /// keeps its original feature tree. No activation crosses the host boundary.
    pub fn project_vector(&self,input:&spark_gpu_probe::driver::DeviceVector<'_>)->Result<spark_gpu_probe::driver::Mxfp8ProjectionPass<'d>> {
        use spark_gpu_probe::{causal_ops::{Shape,Op},driver::NormOutput};
        let b=self.backend;b.require_resident_stream()?;
        if input.len()!=self.source.cols() || input.is_segmented() {return Err("dynamic MXFP8 input extent differs".into());}
        let s=Shape {op:if self.positive_zero {Op::Mxfp8QuantizePositiveZero} else {Op::Mxfp8Quantize},heads:0,dim:0,taps:0,rows:0,cols:input.len().try_into().map_err(|_|"MXFP8 columns exceed ABI")?,epsilon:0.0,bound:0.0};
        let decoded=b.causal_operation(s,[Some(input),None,None,None],None)?;
        self.projection.apply(&NormOutput::F32(decoded))
    }
    /// Decoded BF16 lanes of many token rows. Output `[tokens,rows]`, rounded.
    pub fn project_rows(&self,input:&spark_gpu_probe::driver::DeviceVector<'_>,tokens:usize)->Result<spark_gpu_probe::driver::Bf16Vector<'d>> {
        let decoded=self.quantize_rows(input,tokens)?;self.project_quantized_rows(&decoded,tokens)
    }
    /// The first half of `project_rows`: the quantized token rows `[tokens,cols]`, decoded. It reads only the rows and
    /// the column count, never a weight, so every MXFP8 matrix of this width that reads the same rows may share it.
    pub fn quantize_rows(&self,input:&spark_gpu_probe::driver::DeviceVector<'_>,tokens:usize)->Result<spark_gpu_probe::driver::DeviceVector<'d>> {
        use spark_gpu_probe::causal_ops::{Shape,Op};
        let b=self.backend;b.require_resident_stream()?;
        if input.len()!=self.source.cols().checked_mul(tokens).ok_or("dynamic MXFP8 rows overflow")? || input.is_segmented() {return Err("dynamic MXFP8 input extent differs".into());}
        // The quantizer owns complete 32-lane groups, so concatenated rows quantize independently.
        let s=Shape {op:Op::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:0,cols:input.len().try_into().map_err(|_|"MXFP8 columns exceed ABI")?,epsilon:0.0,bound:0.0};
        b.causal_operation(s,[Some(input),None,None,None],None)
    }
    /// The second half of `project_rows`, on the rows `quantize_rows` made for this width.
    pub fn project_quantized_rows(&self,decoded:&spark_gpu_probe::driver::DeviceVector<'_>,tokens:usize)->Result<spark_gpu_probe::driver::Bf16Vector<'d>> {
        let b=self.backend;b.require_resident_stream()?;
        if decoded.len()!=self.source.cols().checked_mul(tokens).ok_or("dynamic MXFP8 rows overflow")? || decoded.is_segmented() {return Err("dynamic MXFP8 quantized extent differs".into());}
        let accumulator=self.projection.apply_rows(decoded,tokens)?;
        b.round_bf16_lanes(&accumulator)
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
        let activation=if self.positive_zero {Activation::from_f32_positive_zero(&values)?} else {Activation::from_f32(&values)?};
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
