//! One fixed latent arithmetic path. The loader only transposes checkpoint bits.
use super::*;
use spark_gpu_probe::causal_ops::{Shape as Causal,Op as CausalOp};
pub(super) struct Absorbed<'d> {shape:Causal,weight:DeviceVector<'d>}
fn spec(c:&CheckpointAdmission,name:&str)->Result<Option<engine_model::plan::description::LatentDescription>> {
    use engine_model::{plan::{glm_index::{TensorRole,LatentTensor},description::TokenMixer},shapes::Place};
    let binding=c.description().bind_tensor(name).map_err(|e|e.to_string())?;
    if binding.operation!=TensorRole::Latent(LatentTensor::KvUp) {return Ok(None);}
    let spec=match binding.role.place {
        Place::Layer(layer)=>c.description().layers.get(layer as usize).map(|l|l.mixer),
        // A draft tensor belongs to the draft layer, which carries its own mixer geometry.
        Place::Draft {..}=>c.description().with_draft_layer().map_err(|e|e.to_string())?.and_then(|d|d.layers.last().map(|l|l.mixer)),
        _=>return Ok(None),
    };
    let Some(TokenMixer::LatentAttention(spec))=spec else {return Err("latent projection has no attention geometry".into());};
    let view=c.headers().weights().get(name).ok_or("latent projection header is absent")?;
    if spec.rotary_dim!=0 || spec.kv_rank%2!=0 || view.components().first().map(|c|c.dtype)!=Some(Dtype::Bf16) || view.logical().storage!=WeightStorage::Plain {return Ok(None);}
    Ok(Some(spec))
}
pub(super) fn rows(c:&CheckpointAdmission,name:&str,topology:Topology)->Result<Option<Rows>> {
    let Some(spec)=spec(c,name)? else {return Ok(None);};
    let layout=crate::sparse_heads::Layout::new(spec.heads,spec.key_dim,spec.value_dim,topology)?;
    let owned=layout.projection_rows();let total=layout.global_row_words();
    Ok(Some(Rows {start:owned.start,count:owned.len(),total,split:owned.len()!=total}))
}
pub(super) fn shape(c:&CheckpointAdmission,name:&str,rows:Rows,cols:usize)->Result<Option<Causal>> {
    let Some(spec)=spec(c,name)? else {return Ok(None);};
    let width=(spec.key_dim+spec.value_dim) as usize;
    if width==0 || !rows.count.is_multiple_of(width) || !rows.start.is_multiple_of(width) || cols!=spec.kv_rank as usize {return Err("latent projection does not own complete heads".into());}
    let number=|v:usize|u32::try_from(v).map_err(|_|"latent dimension exceeds ABI".to_string());
    Ok(Some(Causal {op:CausalOp::AbsorbQuery,heads:number(rows.count/width)?,dim:number(spec.key_dim as usize)?,taps:number(spec.value_dim as usize)?,rows:1,cols:number(cols)?,epsilon:0.0,bound:0.0}))
}
pub(super) fn load<'d>(b:&Backend<'d>,shape:Causal,bytes:&[u8])->Result<Absorbed<'d>> {
    let (h,d,v,k)=(shape.heads as usize,shape.dim as usize,shape.taps as usize,shape.cols as usize);
    if bytes.len()!=h*(d+v)*k*2 {return Err("absorbed checkpoint extent differs".into());}
    let source=bytes.as_chunks::<2>().0;let mut target=source.to_vec();
    for head in 0..h {for j in 0..k {for feature in 0..d {target[head*(d+v)*k+j*d+feature]=source[(head*(d+v)+feature)*k+j];}}}
    let packed=target.as_chunks::<2>().0.iter().map(|pair|f32::from_bits(u16::from_le_bytes(pair[0]) as u32|(u16::from_le_bytes(pair[1]) as u32)<<16)).collect::<Vec<_>>();
    let weight=b.upload_vector(&packed)?;
    BF16_COPY_BYTES.fetch_add(weight.len()*4,std::sync::atomic::Ordering::Relaxed);BF16_COPY_MATRICES.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
    Ok(Absorbed {shape,weight})
}
impl<'b,'d> Weights<'b,'d> {
    pub fn absorb_rows(&self,name:&str,b:&Backend<'d>,shape:spark_gpu_probe::prompt_ops::Shape,query:&DeviceVector<'_>,latent:&DeviceVector<'_>,selected:&DeviceVector<'_>)->Result<DeviceVector<'d>> {
        use spark_gpu_probe::prompt_ops::{Op,Shape};
        let a=self.matrices.get(name).and_then(|m|m.absorbed.as_ref()).ok_or("absorbed projection is not resident")?;
        if (shape.heads,shape.dim,shape.taps,shape.cols)!=(a.shape.heads,a.shape.dim,a.shape.taps,a.shape.cols) {return Err("absorbed prompt geometry differs".into());}
        let q=b.prompt_operation(Shape {op:Op::AbsorbQuery,..shape},[Some(query),None,None,None,None],Some(&a.weight))?;
        let scores=b.prompt_operation(Shape {op:Op::AbsorbScores,..shape},[Some(&q),Some(latent),Some(selected),None,None],None)?;
        let probability=b.prompt_operation(Shape {op:Op::AbsorbSoftmax,cols:shape.first,..shape},[Some(&scores),Some(selected),None,None,None],None)?;
        let values=b.prompt_operation(Shape {op:Op::AbsorbValues,..shape},[Some(&probability),Some(latent),Some(selected),None,None],None)?;
        b.prompt_operation(Shape {op:Op::AbsorbProject,..shape},[Some(&values),Some(selected),None,None,None],Some(&a.weight))
    }
    pub fn has_absorption(&self,name:&str)->bool {self.matrices.get(name).is_some_and(|m|m.absorbed.is_some())}
    /// `slots` is the layer's row bound. Scores and probabilities are heads x rows words; their storage is heads x slots
    /// words (the words past heads x rows are never read), so every decode step takes the same vector sizes.
    pub fn absorb(&self,name:&str,b:&Backend<'d>,shape:Causal,query:&DeviceVector<'_>,latent:&DeviceVector<'_>,count:&DeviceVector<'_>,slots:usize)->Result<DeviceVector<'d>> {
        let a=self.matrices.get(name).and_then(|m|m.absorbed.as_ref()).ok_or("absorbed projection is not resident")?;
        if (shape.heads,shape.dim,shape.taps,shape.cols)!=(a.shape.heads,a.shape.dim,a.shape.taps,a.shape.cols) {return Err("absorbed projection geometry differs".into());}
        let q=b.causal_operation(Causal {op:CausalOp::AbsorbQuery,..shape},[Some(query),None,None,None],Some(&a.weight))?;
        if (shape.rows as usize)>slots {return Err("absorbed rows exceed the layer's slots".into());}
        let held=|s:Causal,input:[Option<&DeviceVector<'_>>;4]|->Result<DeviceVector<'d>> {
            let (_,_,units,words)=s.lengths().ok_or("invalid absorbed shape")?;let len=units*words;
            let whole=b.vector((s.heads as usize).checked_mul(slots).ok_or("absorbed slots overflow")?.max(len))?.into_resident_shared()?;
            let mut output=whole.resident_range(0..len)?;b.resident_causal_into(s,input,None,&mut output)?;Ok(output)
        };
        let scores=held(Causal {op:CausalOp::AbsorbScores,..shape},[Some(&q),Some(latent),Some(count),None])?;
        let probability=held(Causal {op:CausalOp::Softmax,taps:0,cols:0,..shape},[Some(&scores),Some(count),None,None])?;
        let values=b.causal_operation(Causal {op:CausalOp::AbsorbValues,..shape},[Some(&probability),Some(latent),Some(count),None],None)?;
        let output=b.causal_operation(Causal {op:CausalOp::AbsorbProject,..shape},[Some(&values),None,None,None],Some(&a.weight));
        #[cfg(feature="kernel-bench")]
        if output.is_ok() {ABSORPTION_CALLS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);}
        output
    }
}
