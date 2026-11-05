//! Checkpoint-bound causal KDA/dense layer. Full service admission stays separate.
use crate::{Result,layer_driver::LayerBoundary,layer_operands::ResidualBlockOperands,mxfp8_projection::Mxfp8Projection};
use engine_model::{plan::{admission::CheckpointAdmission,description::TokenMixer,layer::{Kind,LayerPlan,Topology},state::StatePlan,storage::WeightStorage},shapes::Ffn,safetensors::Dtype};
use spark_gpu_probe::{causal_ops::{self,Shape,Op},mhc_boundary,mhc_reference,norm_boundary::NormShape};
use std::collections::BTreeMap;
pub type Planes=BTreeMap<String,Vec<f32>>;
pub struct State {pub history:Vec<f32>,pub recurrent:Vec<f32>,pub position:u64}
pub(crate) enum Prepared<'f> {Residual(ResidualBlockOperands),Projection(Mxfp8Projection<'f>),Plain {shape:Shape,weight:Vec<f32>},
    Operation {shape:Shape,weight:Vec<f32>},Combine,Skip}
pub struct Layer<'f> {pub(crate) plan:LayerPlan,pub(crate) operations:Vec<Prepared<'f>>}
fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn plain(c:&CheckpointAdmission,name:&str)->Result<Vec<f32>> {
    let u=c.upload(name).map_err(|e|e.to_string())?;
    need(u.storage==WeightStorage::Plain,"causal parameter requires plain storage")?;
    let values:Vec<_>=match u.dtype {
        Dtype::Bf16=>u.weight.chunks_exact(2).map(|b|f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap()))<<16)).collect(),
        Dtype::F32=>u.weight.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect(),
        _=>return Err("causal parameter requires BF16 or FP32 values".into()),
    };need(values.iter().all(|v|v.is_finite()),"causal parameter is not finite")?;Ok(values)
}
fn base(op:Op,heads:u64,dim:u64,taps:u64,epsilon:f64,bound:f64)->Result<Shape> {
    let s=Shape {op,heads:heads.try_into().map_err(|_|"causal head count exceeds ABI")?,dim:dim.try_into().map_err(|_|"causal dimension exceeds ABI")?,
        taps:taps.try_into().map_err(|_|"causal tap count exceeds ABI")?,rows:0,cols:0,epsilon:epsilon as f32,bound:bound as f32};Ok(s)
}
impl<'f> Layer<'f> {
    /// Prepare only complete KDA layers with dense feed-forward recipes.
    /// Refuse other layer kinds before payload reads or device allocation.
    pub fn new(c:&'f CheckpointAdmission,layer:usize)->Result<Self> {
        let d=c.description();let block=d.layers.get(layer).ok_or("causal layer index is outside the decoder")?;
        let TokenMixer::Kda(k)=block.mixer else {return Err("causal layer requires a KDA recipe".into());};
        need(matches!(block.ffn,Ffn::Dense {..}),"causal dense layer cannot admit an expert recipe")?;
        let topology=Topology {rank:0,parts:1};let state=StatePlan::new(d,topology,1,isize::MAX as usize).map_err(|e|e.to_string())?;
        let boundary=LayerBoundary::new(d,c.headers(),&state,layer,1,topology,usize::MAX)?;
        let plan=boundary.program().clone();let mut operations=Vec::new();
        for (i,step) in plan.steps.iter().enumerate() {
            let prepared=match step.kind {
                Kind::MhcPrepare=>Prepared::Residual(boundary.prepare_residual_block_checkpoint(i,c)?),
                Kind::Collapse|Kind::Norm=>Prepared::Skip,
                Kind::Combine=>Prepared::Combine,
                Kind::Project if step.weights[0].storage==WeightStorage::Mxfp8=>Prepared::Projection(boundary.prepare_mxfp8_projection_checkpoint(i,c)?),
                Kind::Project=>{
                    let w=&step.weights[0];need(w.storage==WeightStorage::Plain && w.shape.len()==2,"causal projection storage is unsupported")?;
                    let mut shape=base(Op::Project,0,0,0,d.epsilon,0.0)?;
                    shape.rows=w.shape[0].try_into().map_err(|_|"causal projection rows exceed ABI")?;
                    shape.cols=w.shape[1].try_into().map_err(|_|"causal projection columns exceed ABI")?;
                    need(shape.lengths().is_some(),"invalid causal projection geometry")?;
                    Prepared::Plain {shape,weight:plain(c,&w.name)?}
                },
                kind=>{
                    let (op,bound)=match kind {Kind::Conv=>(Op::Conv,0.0),Kind::KdaPrepare=>(Op::Prepare,k.gate_lower_bound),
                        Kind::KdaStep=>(Op::Step,0.0),Kind::GatedNorm=>(Op::GatedNorm,0.0),Kind::SwiGlu=>(Op::SwiGlu,d.swiglu_limit),
                        _=>return Err(format!("{}: causal operation is unsupported",step.id))};
                    let mut shape=base(op,k.heads,k.head_dim,k.conv,d.epsilon,bound)?;
                    if op==Op::SwiGlu {shape.rows=step.outputs[0].elements().map_err(|e|e.to_string())?.try_into().map_err(|_|"causal middle exceeds ABI")?;}
                    let mut weight=Vec::new();for w in &step.weights {weight.extend(plain(c,&w.name)?);}
                    let (_,expected,_,_)=shape.lengths().ok_or("invalid causal operation shape")?;
                    need(weight.len()==expected,"causal parameter count differs from recipe")?;
                    Prepared::Operation {shape,weight}
                },
            };operations.push(prepared);
        }
        Ok(Self {plan,operations})
    }
    pub fn plan(&self)->&LayerPlan {&self.plan}
    pub fn zero_state(&self)->Result<State> {
        let TokenMixer::Kda(k)=self.plan.settings.decoder.mixer else {return Err("causal state requires KDA".into());};
        let history=self.plan.inputs.iter().find(|b|b.name=="kda.history").ok_or("causal history is absent")?.elements().map_err(|e|e.to_string())?;
        let state=self.plan.inputs.iter().find(|b|b.name=="kda.state").ok_or("causal recurrence is absent")?.elements().map_err(|e|e.to_string())?;
        need(k.conv>0,"causal history has no taps")?;Ok(State {history:vec![0.0;history],recurrent:vec![0.0;state],position:0})
    }
    pub fn reference(&self,residual:&[f32],state:&State)->Result<(Planes,State)> {
        let mut values=Planes::new();values.insert("residual".into(),residual.to_vec());
        values.insert("kda.history".into(),state.history.clone());values.insert("kda.state".into(),state.recurrent.clone());
        for (step,prepared) in self.plan.steps.iter().zip(&self.operations) {
            let inputs=step.inputs.iter().map(|n|values.get(n).map(Vec::as_slice).ok_or_else(||format!("missing causal input {n}"))).collect::<Result<Vec<_>>>()?;
            let output=match prepared {
                Prepared::Residual(p)=>{
                    let m=p.mhc();let s=m.shape();need(inputs[0].len()==s.wide(),"causal residual extent differs")?;
                    mhc_boundary::check_bf16(inputs[0]).map_err(str::to_owned)?;
                    let mut normalized=vec![0.0;s.wide()];let mut coeff=vec![0.0;s.coefficients()];
                    mhc_reference::prepare(s,inputs[0],m.weights(),m.base(),m.scale(),&mut normalized,&mut coeff);
                    let collapsed=(0..s.hidden as usize).map(|j|mhc_boundary::collapse(inputs[0],&coeff,s.streams as usize,s.hidden as usize,j)).collect::<Vec<_>>();
                    let norm=norm(p.norm().shape(),&collapsed,p.norm().weight());let side=step.id.split('.').next().unwrap();
                    vec![(format!("{side}.normalized"),normalized),(format!("{side}.coefficients"),coeff),(format!("{side}.collapsed"),collapsed),
                        (if side=="attn" {"input_norm"} else {"post_norm"}.into(),norm)]
                },
                Prepared::Projection(p)=>vec![(step.outputs[0].name.clone(),p.dynamic_reference(inputs[0])?.2)],
                Prepared::Plain {shape,weight}=>{
                    let v=causal_ops::reference(*shape,[inputs[0],&[],&[],&[]],weight)?;
                    vec![(step.outputs[0].name.clone(),v.into_iter().map(spark_gpu_probe::bf16::rounded).collect())]
                },
                Prepared::Operation {shape,weight}=>{
                    let x=operation_inputs(*shape,&inputs)?;let packed=causal_ops::reference(*shape,x.each_ref().map(Vec::as_slice),weight)?;
                    unpack(*shape,&packed)?.into_iter().zip(&step.outputs).map(|(v,b)|(b.name.clone(),v)).collect()
                },
                Prepared::Combine=>{
                    let s=self.plan.settings.residual.streams as usize;let h=self.plan.settings.hidden as usize;
                    let v=(0..s*h).map(|j|mhc_boundary::combine(inputs[0],inputs[1],inputs[2][j%h],s,h,j/h,j%h)).collect();
                    vec![(step.outputs[0].name.clone(),v)]
                },
                Prepared::Skip=>continue,
            };
            for (name,value) in output {need(value.iter().all(|v|v.is_finite()),"causal layer output is not finite")?;values.insert(name,value);}
        }
        let next=State {history:values["next_history"].clone(),recurrent:values["next_state"].clone(),position:state.position.checked_add(1).ok_or("causal position overflow")?};
        Ok((values,next))
    }
}
pub(crate) fn operation_inputs(shape:Shape,x:&[&[f32]])->Result<[Vec<f32>;4]> {
    let mut input:[Vec<f32>;4]=std::array::from_fn(|_|Vec::new());
    if shape.op==Op::Conv {need(x.len()==4,"causal convolution input count differs")?;input[0]=x[..3].concat();input[1]=x[3].to_vec();}
    else {need(x.len()<=4,"causal operation input count differs")?;for (a,b) in input.iter_mut().zip(x) {*a=b.to_vec();}}
    Ok(input)
}
pub(crate) fn unpack(s:Shape,packed:&[f32])->Result<Vec<Vec<f32>>> {
    let (_,_,units,words)=s.lengths().ok_or("invalid causal output shape")?;need(packed.len()==units*words,"causal packed output extent differs")?;
    let (h,d)=(s.heads as usize,s.dim as usize);let width=h*d;
    Ok(match s.op {
        Op::Conv=>{let mut out=Vec::new();let mut history=Vec::new();for plane in packed.chunks_exact(words) {out.extend_from_slice(&plane[..width]);history.extend_from_slice(&plane[width..]);}vec![out,history]},
        Op::Step=>{let mut state=Vec::new();let mut out=Vec::new();for head in packed.chunks_exact(words) {state.extend_from_slice(&head[..d*d]);out.extend_from_slice(&head[d*d..]);}vec![out,state]},
        _=>vec![packed.to_vec()],
    })
}
/// Match the existing resident ordinary norm tree and stored parameter precision.
pub fn norm(s:NormShape,x:&[f32],weight:&[f32])->Vec<f32> {
    fn sum(mut lanes:[f32;32])->f32 {for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32 {lanes[lane]=old[lane]+old[lane^distance];}}lanes[0]}
    let partials=x.chunks(128).map(|chunk|{let mut lanes=[0.0;32];for lane in 0..32 {let mut i=lane;while i<chunk.len() {lanes[lane]+=chunk[i]*chunk[i];i+=32;}}sum(lanes)}).collect::<Vec<_>>();
    let mut lanes=[0.0;32];for lane in 0..32 {let mut i=lane;while i<partials.len() {lanes[lane]+=partials[i];i+=32;}}
    let scale=1.0/(sum(lanes)/s.width() as f32+s.epsilon()).sqrt();x.iter().zip(weight).map(|(&x,&w)|s.weight(x*scale,w)).collect()
}
#[cfg(target_os="linux")]
#[path="causal_layer/device.rs"] mod device;
#[cfg(target_os="linux")]
pub use device::{ResidentLayer,DeviceState};
impl Layer<'_> {
    /// Pin every consumed parameter. The caller supplies its receipt hash.
    pub fn parameter_record(&self,mut digest:impl FnMut(&[u8])->String)->Result<String> {
        let mut text=format!("layer={}\nsettings={:?}\n",self.plan.layer,self.plan.settings);
        let bytes=|v:&[f32]|v.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
        for (step,p) in self.plan.steps.iter().zip(&self.operations) {
            text.push_str(&format!("step={} kind={:?} weights={:?}\n",step.id,step.kind,step.weights));
            match p {
                Prepared::Residual(p)=>{let m=p.mhc();text.push_str(&format!("mhc_shape={:?} mhc_dtype={:?} norm_shape={:?} norm_dtype={:?}\n",m.shape(),m.stored_dtypes(),p.norm().shape(),p.norm().stored_dtype()));
                    for v in [m.weights(),m.base(),m.scale(),p.norm().weight()] {text.push_str(&format!("parameter={}\n",digest(&bytes(v))));}},
                Prepared::Projection(p)=>{let (w,s)=p.read()?;text.push_str(&format!("mx_weight={} mx_scale={}\n",digest(&w),digest(&s)));},
                Prepared::Plain {shape,weight}|Prepared::Operation {shape,weight}=>text.push_str(&format!("shape={shape:?} parameter={}\n",digest(&bytes(weight)))),
                _=>{},
            }
        }Ok(text)
    }
}
