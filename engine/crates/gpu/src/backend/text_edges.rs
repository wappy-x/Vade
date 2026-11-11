//! Resident input/output edges. Every call completes on the owning stream.
use super::*;
use crate::text_edges::{Layout, Shape, NormStorage};

pub struct ResidentTextEdges<'b,'a> {
    backend: &'b Backend<'a>, shape: Shape, embedding: Matrix<'a>, head: Buffer<'a>, scales: Buffer<'a>,
    norm: DeviceVector<'a>, norm_storage: NormStorage, epsilon: f32,
}
impl<'a> Backend<'a> {
    fn text_edge_ready(&self) -> Result<(),String> {
        if self.context.capturing.get() {return Err("text edges require an uncaptured stream".into());}
        if let Some(error) = self.fatal_error() {return Err(error);}
        Ok(())
    }
    /// Preserve checkpoint weight and scale bytes. This checks physical spans;
    /// the model loader must reconcile the original tensor headers and recipes.
    pub fn resident_text_edges(&self, arena: &WeightArena<'a>, layout: Layout) -> Result<ResidentTextEdges<'_,'a>,String> {
        self.text_edge_ready()?;
        if !ptr::eq(arena.buffer._context,self.context) {return Err("text weights belong to another context".into());}
        let ranges = layout.ranges(arena.bytes())?; let s = layout.shape;
        let embedding = self.weight_matrix(arena,layout.embedding,s.vocab(),s.hidden(),crate::load::WeightDtype::Bf16)?;
        let head = Buffer::view(&arena.buffer,ranges[1].start,ranges[1].len())?;
        let scales = Buffer::view(&arena.buffer,ranges[2].start,ranges[2].len())?;
        // Small norm parameters widen losslessly. No BF16 conversion of stored
        // FP32 values is permitted. The source header selects the representation.
        let mut stored = vec![0; ranges[3].len()]; arena.read(ranges[3].start,&mut stored)?;
        let values:Vec<f32> = match layout.norm_storage {
            NormStorage::Bf16 => stored.as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b))<<16)).collect(),
            NormStorage::F32 => stored.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        };
        if values.iter().any(|x| !x.is_finite()) {return Err("text norm weight is not finite".into());}
        let norm = self.upload_vector(&values)?;
        self.require_arm_kernels(&["mhc_head","flash_next_mxfp8_m1"])?;
        Ok(ResidentTextEdges {backend:self,shape:s,embedding,head,scales,norm,norm_storage:layout.norm_storage,epsilon:layout.epsilon})
    }
}
impl<'a> ResidentTextEdges<'_,'a> {
    pub fn shape(&self) -> Shape {self.shape}
    /// Repeat the token ID across streams. The embedding kernel copies each
    /// stored BF16 row into FP32 lanes without arithmetic or a fixed stream count.
    pub fn embed(&self, token: u32) -> Result<DeviceVector<'a>,String> {
        let b = self.backend; let s = self.shape; b.text_edge_ready()?; s.token(token)?;
        let ids = b.upload_vector(&vec![token as f32;s.streams()])?;
        let mut output = b.vector(s.residual())?;
        let launched = b.batch_embedding_device(&self.embedding,&ids,&mut output);
        let completed = b.synchronize(); launched.and(completed)?; Ok(output)
    }
    /// Complete the checked stream mean on the owning backend.
    pub fn mean(&self,streams:&DeviceVector<'_>)->Result<DeviceVector<'a>,String> {
        self.backend.text_mean(self.shape,streams)
    }
    /// Normalize a BF16 mean row in FP32 lanes. The fixed feature reduction
    /// rounds the normalized value before multiplying by the original weight.
    /// A stored BF16 weight gives a BF16 product. A stored FP32 weight keeps its
    /// FP32 product. The loaded-dtype policy and reduction still need qualification.
    pub fn norm(&self, mean: &DeviceVector<'_>) -> Result<DeviceVector<'a>,String> {
        let b = self.backend; let h = self.shape.hidden(); b.text_edge_ready()?;
        b.check_vector(mean,h)?;
        if mean.is_segmented() {return Err("text norm input must be contiguous".into());}
        let mut normalized = b.vector(h)?; let mut partials = b.vector(h.div_ceil(128))?;
        let applied = b.model_rms_device(mean,None,&mut normalized,&mut partials,h,1,self.epsilon);
        let completed = b.synchronize(); applied.and(completed)?;
        let weighted = b.model_elementwise_device(&normalized,Some(&self.norm),&normalized,h,4,1.0);
        let completed = b.synchronize(); weighted.and(completed)?;
        if self.norm_storage == NormStorage::Bf16 {
            let rounded = b.model_elementwise_device(&normalized,None,&normalized,h,0,1.0);
            let completed = b.synchronize(); rounded.and(completed)?;
        }
        Ok(normalized)
    }
    /// Project one already-normalized FP32 row through the original stored head.
    /// No input quantizer, output cast, or model admission is implied.
    /// Each call uses the same single-row kernel and complete input dimension.
    pub fn project_fp32(&self, normalized: &DeviceVector<'_>) -> Result<DeviceVector<'a>,String> {
        let b = self.backend; let s = self.shape; b.text_edge_ready()?;
        b.check_vector(normalized,s.hidden())?;
        if normalized.is_segmented() {return Err("text head input must be contiguous".into());}
        let output = b.vector(s.vocab())?;
        let mut args = crate::next_params::MxLayer::default();
        args.push(crate::next_params::MxProjection {weights:self.head.address,scales:self.scales.address,
            input:normalized.device_address(),output:output.device_address(),rows:s.vocab() as u32,
            cols:s.hidden() as u32,tokens:1,input_bf16:0,round:0})?;
        // SAFETY: Layout checked complete E4M3/UE8M0 spans and weight alignment.
        // The private views retain their arena. Shape and check_vector bound the
        // input; MxLayer checks the ABI and alignment. The new output is disjoint.
        // All owners and the by-value argument live through stream completion.
        let launched = unsafe {b.launch_flash("flash_next_mxfp8_m1",args.ends[1],128,&mut [parameter(&mut args)])};
        let completed = b.synchronize(); launched.and(completed)?; Ok(output)
    }
}

impl<'a> Backend<'a> {
    /// Fixed ascending stream mean, then RN-even BF16 in FP32 lanes. This is a
    /// candidate reduction contract. It still needs the checkpoint device gate.
    pub fn text_mean(&self, s: Shape, streams: &DeviceVector<'_>) -> Result<DeviceVector<'a>,String> {
        let b=self;b.text_edge_ready()?;b.require_arm_kernels(&["mhc_head"])?;
        b.check_vector(streams,s.residual())?;
        if streams.is_segmented() {return Err("text streams must be contiguous".into());}
        let output = b.vector(s.hidden())?;
        let mut args = crate::mhc_params::Head {input:streams.device_address(),output:output.device_address(),
            rows:1,streams:s.streams() as u32,hidden:s.hidden() as u32};
        // SAFETY: Shape bounds the input and output extents. check_vector checks
        // the input context and exact length. Output is a new disjoint allocation.
        // Both owners and the by-value argument live through stream completion.
        let launched = unsafe {b.launch_flash("mhc_head",(s.hidden() as u32).div_ceil(128),128,&mut [parameter(&mut args)])};
        let completed = b.synchronize(); launched.and(completed)?;
        let rounded = b.model_elementwise_device(&output,None,&output,s.hidden(),0,1.0);
        let completed = b.synchronize(); rounded.and(completed)?; Ok(output)
    }
}
