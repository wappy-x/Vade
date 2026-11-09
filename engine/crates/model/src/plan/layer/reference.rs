//! Scalar logical FP32 layer reference. No checkpoint casts or device calls.
//! The schedule below is independent of LayerPlan::new. Events use computed shapes.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Data { F32(Vec<f32>), U32(Vec<u32>) }
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor { pub shape: Vec<u64>, pub data: Data }
impl Tensor {
    pub fn f32(shape: &[u64], values: Vec<f32>) -> Result<Self> {
        let b=Buffer::new("host",shape);
        need(b.elements()? == values.len() && values.iter().all(|v| v.is_finite()),"invalid host tensor")?;
        Ok(Self { shape: shape.into(),data: Data::F32(values) })
    }
    pub fn floats(&self) -> Result<&[f32]> { match &self.data { Data::F32(v) => Ok(v),_ => Err(invalid("expected FP32 host tensor")) } }
    fn ids(&self) -> Result<&[u32]> { match &self.data { Data::U32(v) => Ok(v),_ => Err(invalid("expected integer host tensor")) } }
    fn spec(&self, name: &str) -> Buffer { Buffer { name: name.into(),shape: self.shape.clone(),element: match self.data { Data::F32(_) => Element::F32,Data::U32(_) => Element::U32 } } }
}
pub type Tensors = BTreeMap<String,Tensor>;
/// Supply decoded host weights on demand. This does not define device casts.
pub trait WeightSource {
    /// Declare value semantics explicitly. A stored dtype is not a runtime profile.
    fn profile(&self) -> profile::Profile;
    fn shape(&self, name: &str) -> Result<Vec<u64>>;
    fn load(&self, name: &str) -> Result<Tensor>;
    /// Read one complete logical matrix row. A checkpoint reader can override
    /// this method to avoid decoding an entire embedding table.
    fn row(&self, name: &str, index: u32) -> Result<Tensor> {
        let t=self.load(name)?;
        need(t.shape.len()==2 && u64::from(index)<t.shape[0],"host weight row is outside matrix")?;
        let width=usize::try_from(t.shape[1]).map_err(|_| invalid("host weight row width overflow"))?;
        let start=(index as usize).checked_mul(width).ok_or_else(|| invalid("host weight row offset overflow"))?;
        let end=start.checked_add(width).ok_or_else(|| invalid("host weight row end overflow"))?;
        Tensor::f32(&[t.shape[1]],t.floats()?.get(start..end).ok_or_else(|| invalid("host weight row extent differs"))?.to_vec())
    }
}
impl WeightSource for Tensors {
    fn profile(&self) -> profile::Profile { profile::Profile::LogicalF32V1 }
    fn shape(&self, name: &str) -> Result<Vec<u64>> { self.get(name).map(|t| t.shape.clone()).ok_or_else(|| invalid(format!("missing host weight {name}"))) }
    fn load(&self, name: &str) -> Result<Tensor> { self.get(name).cloned().ok_or_else(|| invalid(format!("missing host weight {name}"))) }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event { pub id: String, pub kind: Kind, pub inputs: Vec<Buffer>, pub outputs: Vec<Buffer>, pub weights: Vec<Buffer> }
#[derive(Debug)]
pub struct Forward { pub tensors: Tensors, pub trace: Vec<Event>, pub settings: Settings, pub layer: usize, pub context: u64 }
impl Forward {
    /// Compare order, names, input/output shapes, types, and logical weight shapes.
    pub fn check_plan(&self, plan: &LayerPlan) -> Result<()> {
        plan.validate()?;
        need(self.settings==plan.settings && self.layer==plan.layer && self.context==plan.context,"host and plan settings differ")?;
        need(self.trace.len() == plan.steps.len(),"host and plan operation counts differ")?;
        let mut buffers: BTreeMap<_,_>=plan.inputs.iter().map(|b| (b.name.clone(),b.clone())).collect();
        for (event,step) in self.trace.iter().zip(&plan.steps) {
            need(event.id == step.id && event.kind == step.kind,"host and plan operation order differs")?;
            let inputs=step.inputs.iter().map(|n| buffers[n].clone()).collect::<Vec<_>>();
            let weights=step.weights.iter().map(|w| Buffer::new(&w.name,&w.shape)).collect::<Vec<_>>();
            need(event.inputs == inputs && event.outputs == step.outputs && event.weights == weights,
                &format!("host and plan shapes differ at {}",step.id))?;
            for b in &step.outputs { buffers.insert(b.name.clone(),b.clone()); }
        }
        Ok(())
    }
}

/// Replay the declared program on the host. This checks the program's data flow.
/// It shares scalar math with the independent forward schedule, not device math.
pub fn replay(d: &TextDescription, plan: &LayerPlan, weights: &Tensors, inputs: &Tensors) -> Result<Forward> {
    plan.validate()?;
    validate_data(d,plan.layer,plan.context)?;
    let settings=Settings::from_description(d,plan.layer)?;
    need(settings==plan.settings,"replay settings differ from plan")?;
    let mut cpu=Cpu { d,layer:plan.layer,context:plan.context,weights,tensors:inputs.clone(),trace:Vec::new() };
    let prefix=format!("model.language_model.layers.{}.",plan.layer);
    for step in &plan.steps {
        let names=step.weights.iter().map(|w| w.name.strip_prefix(&prefix).ok_or_else(|| invalid("weight is outside layer"))).collect::<Result<Vec<_>>>()?;
        cpu.call(&step.id,step.kind,&step.inputs.iter().map(String::as_str).collect::<Vec<_>>(),&names)?;
    }
    let out=Forward { tensors:cpu.tensors,trace:cpu.trace,settings,layer:plan.layer,context:plan.context }; out.check_plan(plan)?; Ok(out)
}

/// Run one valid token with supplied decoded FP32 weights and earlier cache rows.
/// Inputs use LayerPlan's public buffer names. The schedule does not read a plan.
/// Cached state is returned in next_* outputs. Errors leave caller data unchanged.
pub fn forward(d: &TextDescription, layer: usize, context: u64, weights: &Tensors, inputs: &Tensors) -> Result<Forward> {
    forward_from(d,layer,context,weights,inputs)
}
/// Run the same scalar schedule with on-demand weight loading.
/// Expert IDs keep their original values in the returned tensors and trace.
pub fn forward_from(d: &TextDescription, layer: usize, context: u64, weights: &dyn WeightSource, inputs: &Tensors) -> Result<Forward> {
    weights.profile().require_scalar().map_err(|e| invalid(e.to_string()))?;
    // Validate operation data without constructing or reading a plan.
    validate_data(d,layer,context)?;
    let mut c=Cpu { d,layer,context,weights,tensors: inputs.clone(),trace: Vec::new() };
    c.hc("attn","residual")?;
    c.call("input_norm",Kind::Norm,&["attn.collapsed"],&["input_layernorm.weight"])?;
    match d.layers[layer].mixer {
        TokenMixer::Kda(_) => {
            for name in ["q","k","v"] { c.proj(name,"input_norm",&format!("self_attn.{name}_proj.weight"))?; }
            c.call("conv",Kind::Conv,&["q","k","v","kda.history"],&["self_attn.q_conv1d.weight","self_attn.k_conv1d.weight","self_attn.v_conv1d.weight"])?;
            c.proj("forget_down","input_norm","self_attn.f_a_proj.weight")?;
            c.proj("forget","forget_down","self_attn.f_b_proj.weight")?;
            c.proj("beta_logits","input_norm","self_attn.b_proj.weight")?;
            c.call("kda.prepare",Kind::KdaPrepare,&["convolved","forget","beta_logits"],&["self_attn.A_log","self_attn.dt_bias"])?;
            c.call("kda.step",Kind::KdaStep,&["prepared","kda.state"],&[])?;
            c.proj("output_gate_down","input_norm","self_attn.g_a_proj.weight")?;
            c.proj("output_gate","output_gate_down","self_attn.g_b_proj.weight")?;
            c.call("gated_norm",Kind::GatedNorm,&["recurrent","output_gate"],&["self_attn.o_norm.weight"])?;
            c.proj("attention","gated","self_attn.o_proj.weight")?;
        },
        TokenMixer::LatentAttention(_) => {
            c.proj("query_down","input_norm","self_attn.q_a_proj.weight")?;
            c.call("query_latent",Kind::Norm,&["query_down"],&["self_attn.q_a_layernorm.weight"])?;
            c.proj("query","query_latent","self_attn.q_b_proj.weight")?;
            c.proj("kv_down","input_norm","self_attn.kv_a_proj_with_mqa.weight")?;
            c.call("kv_latent",Kind::Norm,&["kv_down"],&["self_attn.kv_a_layernorm.weight"])?;
            c.call("cache_append",Kind::CacheAppend,&["latent_cache","kv_latent"],&[])?;
            c.proj("index_query","query_latent","self_attn.indexer.wq_b.weight")?;
            c.proj("index_key_raw","input_norm","self_attn.indexer.wk.weight")?;
            c.call("index_key",Kind::KeyNorm,&["index_key_raw"],&["self_attn.indexer.k_norm.weight","self_attn.indexer.k_norm.bias"])?;
            c.proj("index_gate","input_norm","self_attn.indexer.index_kpool_compress_gate")?;
            c.call("pool",Kind::Pool,&["index_keys","index_gates","index_key","index_gate"],&["self_attn.indexer.index_kpool_compress_ape"])?;
            c.proj("index_head_weights","input_norm","self_attn.indexer.weights_proj.weight")?;
            c.call("pool_scores",Kind::ScorePools,&["index_query","pool_keys","index_head_weights"],&[])?;
            c.call("select",Kind::Select,&["pool_scores"],&[])?;
            c.call("gather",Kind::Gather,&["next_latent_cache","selected","selected_count"],&[])?;
            c.call("kv_expand",Kind::ExpandKv,&["selected_latent","selected_count"],&["self_attn.kv_b_proj.weight"])?;
            c.call("scores",Kind::Scores,&["query","keys","selected_count","local_slots"],&[])?;
            c.call("softmax",Kind::Softmax,&["scores","selected_count"],&[])?;
            c.call("values",Kind::Values,&["probabilities","values","selected_count","local_slots"],&[])?;
            c.proj("attention","attended","self_attn.o_proj.weight")?;
        },
    }
    c.call("attn.combine",Kind::Combine,&["residual","attn.coefficients","attention"],&[])?;
    c.hc("ffn","after_attention")?;
    c.call("post_norm",Kind::Norm,&["ffn.collapsed"],&["post_attention_layernorm.weight"])?;
    match d.layers[layer].ffn {
        Ffn::Dense { .. } => c.ffn("dense","mlp","ffn_output")?,
        Ffn::Experts(e) => {
            c.proj("router_logits","post_norm","mlp.gate.weight")?;
            c.call("route",Kind::Route,&["router_logits"],&["mlp.gate.e_score_correction_bias"])?;
            for part in ["gate","up"] {
                let names=(0..e.count).map(|i| format!("mlp.experts.{i}.{part}_proj.weight")).collect::<Vec<_>>();
                c.call(&format!("expert_{part}"),Kind::ExpertProject,&["post_norm","expert_ids"],&names.iter().map(String::as_str).collect::<Vec<_>>())?;
            }
            c.call("expert_middle",Kind::SwiGlu,&["expert_gate","expert_up"],&[])?;
            let names=(0..e.count).map(|i| format!("mlp.experts.{i}.down_proj.weight")).collect::<Vec<_>>();
            c.call("expert_down",Kind::ExpertDown,&["expert_middle","expert_ids"],&names.iter().map(String::as_str).collect::<Vec<_>>())?;
            if e.shared_width > 0 { c.ffn("shared","mlp.shared_experts","shared_output")?; }
            let inputs=if e.shared_width > 0 { vec!["expert_ids","expert_weights","expert_down","shared_output"] }
                else { vec!["expert_ids","expert_weights","expert_down"] };
            c.call("ffn_output",Kind::ExpertSum,&inputs,&[])?;
        },
    }
    c.call("ffn.combine",Kind::Combine,&["after_attention","ffn.coefficients","ffn_output"],&[])?;
    Ok(Forward { tensors: c.tensors,trace: c.trace,settings:Settings::from_description(d,layer)?,layer,context })
}

struct Cpu<'a> { d: &'a TextDescription, layer: usize, context: u64, weights: &'a dyn WeightSource, tensors: Tensors, trace: Vec<Event> }
impl Cpu<'_> {
    fn proj(&mut self,id: &str,input: &str,weight: &str) -> Result<()> { self.call(id,Kind::Project,&[input],&[weight]) }
    fn hc(&mut self,side: &str,input: &str) -> Result<()> {
        self.call(&format!("{side}.prepare"),Kind::MhcPrepare,&[input],&[&format!("hc_{side}_fn"),&format!("hc_{side}_base"),&format!("hc_{side}_scale")])?;
        self.call(&format!("{side}.collapse"),Kind::Collapse,&[input,&format!("{side}.coefficients")],&[])
    }
    fn ffn(&mut self,id: &str,module: &str,output: &str) -> Result<()> {
        self.proj(&format!("{id}.gate"),"post_norm",&format!("{module}.gate_proj.weight"))?;
        self.proj(&format!("{id}.up"),"post_norm",&format!("{module}.up_proj.weight"))?;
        self.call(&format!("{id}.middle"),Kind::SwiGlu,&[&format!("{id}.gate"),&format!("{id}.up")],&[])?;
        self.proj(output,&format!("{id}.middle"),&format!("{module}.down_proj.weight"))
    }
    fn call(&mut self,id: &str,kind: Kind,inputs: &[&str],weights: &[&str]) -> Result<()> {
        let (input_count,weight_count,_)=kind.arity(self.d.layers[self.layer].ffn)?;
        need(inputs.len()==input_count && weights.len()==weight_count,"host operation arity differs")?;
        let input=inputs.iter().map(|&n| self.tensors.get(n).cloned().ok_or_else(|| invalid(format!("missing host input {n}")))).collect::<Result<Vec<_>>>()?;
        let names=weights.iter().map(|n| format!("model.language_model.layers.{}.{n}",self.layer)).collect::<Vec<_>>();
        let shapes=names.iter().map(|n| self.weights.shape(n)).collect::<Result<Vec<_>>>()?;
        let selected=if matches!(kind,Kind::ExpertProject | Kind::ExpertDown) { Some(input[1].ids()?.to_vec()) } else { None };
        let loaded=if let Some(ids)=&selected { ids.iter().map(|&j| names.get(j as usize).ok_or_else(|| invalid("host expert ID is outside weights")))
            .collect::<Result<Vec<_>>>()? } else { names.iter().collect() };
        let w=loaded.iter().map(|n| self.weights.load(n)).collect::<Result<Vec<_>>>()?;
        for (name,tensor) in loaded.iter().zip(&w) { need(tensor.shape==self.weights.shape(name)?,"host weight source shape differs")?; }
        for tensor in input.iter().chain(&w) {
            let size=Buffer::new("host",&tensor.shape).elements()?;
            match &tensor.data { Data::F32(v) => need(v.len()==size && v.iter().all(|x| x.is_finite()),"invalid host FP32 extent")?,
                Data::U32(v) => need(v.len()==size,"invalid host integer extent")? }
        }
        let output=if let Some(ids)=selected { let mut local=input.clone();
            local[1].data=Data::U32((0..ids.len() as u32).collect()); self.compute(id,kind,&local,&w)?
        } else { self.compute(id,kind,&input,&w)? };
        self.trace.push(Event { id:id.into(),kind,inputs: input.iter().zip(inputs).map(|(t,n)| t.spec(n)).collect(),
            outputs: output.iter().map(|(n,t)| t.spec(n)).collect(),weights:names.iter().zip(shapes).map(|(n,shape)| Buffer::new(n,&shape)).collect() });
        for (name,tensor) in output { self.tensors.insert(name,tensor); } Ok(())
    }
    fn compute(&self,id: &str,kind: Kind,x: &[Tensor],w: &[Tensor]) -> Result<Vec<(String,Tensor)>> {
        let (n,h)=(self.d.residual.streams as usize,self.d.hidden as usize);
        let f=x.iter().map(|t| match &t.data { Data::F32(v)=>v.as_slice(),_=>&[] }).collect::<Vec<_>>();
        let wf=w.iter().map(Tensor::floats).collect::<Result<Vec<_>>>()?;
        let one=|name: &str,shape: &[u64],data: Vec<f32>| -> Result<Vec<(String,Tensor)>> { Ok(vec![(name.into(),Tensor::f32(shape,data)?)]) };
        let eps=self.d.epsilon as f32;
        match kind {
            Kind::Project => { let out=project(&x[0],&w[0])?; Ok(vec![(id.into(),out)]) },
            Kind::Norm => one(id,&x[0].shape,rms(f[0],wf[0],eps)?),
            Kind::MhcPrepare => {
                need(f[0].len()==n*h && wf[1].len()==n*(n+2) && wf[2].len()==3,"invalid host mHC shape")?;
                let normalized=rms(f[0],&vec![1.0;n*h],eps)?;
                let mut coeff=project(&Tensor::f32(&[(n*h) as u64],normalized)?,&w[0])?.floats()?.to_vec();
                need(coeff.len()==n*(n+2),"invalid host mHC projection")?;
                for j in 0..coeff.len() { coeff[j]=coeff[j]*wf[2][if j<n {0} else if j<2*n {1} else {2}]+wf[1][j]; }
                let e=self.d.residual.epsilon as f32;
                for v in &mut coeff[..n] { *v=sigmoid(*v)+e; }
                for v in &mut coeff[n..2*n] { *v=2.0*sigmoid(*v); }
                for row in coeff[2*n..].chunks_exact_mut(n) { softmax(row); for v in row { *v+=e; } }
                for iteration in 0..self.d.residual.iterations {
                    if iteration>0 { for row in coeff[2*n..].chunks_exact_mut(n) { let sum=fold(row)+e; for v in row { *v/=sum; } } }
                    for dst in 0..n { let mut sum=0.0; for src in 0..n { sum+=coeff[2*n+src*n+dst]; }
                        for src in 0..n { coeff[2*n+src*n+dst]/=sum+e; } }
                }
                one(&format!("{}.coefficients",id.split('.').next().unwrap()),&[coeff.len() as u64],coeff)
            },
            Kind::Collapse => {
                need(f[0].len()==n*h && f[1].len()==n*(n+2),"invalid host collapse")?;
                let mut out=vec![0.0;h]; for s in 0..n { for j in 0..h { out[j]+=f[1][s]*f[0][s*h+j]; } }
                one(&format!("{}.collapsed",id.split('.').next().unwrap()),&[h as u64],out)
            },
            Kind::Combine => {
                need(f[0].len()==n*h && f[1].len()==n*(n+2) && f[2].len()==h,"invalid host combine")?;
                let mut out=vec![0.0;n*h]; for dst in 0..n { for j in 0..h { let mut sum=0.0;
                    for src in 0..n { sum+=f[1][2*n+src*n+dst]*f[0][src*h+j]; }
                    out[dst*h+j]=f[1][n+dst]*f[2][j]+sum; } }
                one(if id=="attn.combine" {"after_attention"} else {"output"},&[n as u64,h as u64],out)
            },
            Kind::Conv | Kind::KdaPrepare | Kind::KdaStep | Kind::GatedNorm => self.kda(kind,&f,&wf),
            Kind::CacheAppend | Kind::KeyNorm | Kind::Pool | Kind::ScorePools | Kind::Select | Kind::Gather
                | Kind::ExpandKv | Kind::Scores | Kind::Softmax | Kind::Values => self.latent(id,kind,x,w),
            Kind::SwiGlu => {
                need(x[0].shape==x[1].shape,"invalid host SwiGLU shape")?; let limit=self.d.swiglu_limit as f32;
                one(id,&x[0].shape,f[0].iter().zip(f[1]).map(|(&g,&u)| { let g=g.min(limit); (g/(1.0+(-g).exp()))*u.clamp(-limit,limit) }).collect())
            },
            Kind::Route => {
                let Ffn::Experts(e)=self.d.layers[self.layer].ffn else { return Err(invalid("host route requires experts")); };
                let r=self.d.routing; let count=e.count as usize; let groups=r.groups as usize; let width=count/groups;
                need(f[0].len()==count && wf[0].len()==count,"invalid host router shape")?;
                let scores=f[0].iter().map(|&v| sigmoid(v)).collect::<Vec<_>>();
                let choice=scores.iter().zip(wf[0]).map(|(&s,&b)| s+b).collect::<Vec<_>>();
                let mut group_scores=vec![0.0;groups];
                for g in 0..groups { let mut ids=(g*width..(g+1)*width).collect::<Vec<_>>(); sort(&mut ids,&choice); group_scores[g]=choice[ids[0]]+choice[ids[1]]; }
                let mut group_ids=(0..groups).collect::<Vec<_>>(); sort(&mut group_ids,&group_scores); group_ids.truncate(r.selected_groups as usize);
                let mut ids=(0..count).filter(|&j| group_ids.contains(&(j/width))).collect::<Vec<_>>(); sort(&mut ids,&choice); ids.truncate(e.selected as usize);
                let mut weights=ids.iter().map(|&j| scores[j]).collect::<Vec<_>>();
                if r.normalize_selected { let sum=fold(&weights)+1e-20; for v in &mut weights { *v/=sum; } }
                for v in &mut weights { *v*=r.scale as f32; }
                Ok(vec![("expert_ids".into(),Tensor { shape:vec![ids.len() as u64],data:Data::U32(ids.into_iter().map(|j| j as u32).collect()) }),
                    ("expert_weights".into(),Tensor::f32(&[weights.len() as u64],weights)?)])
            },
            Kind::ExpertProject | Kind::ExpertDown => {
                let ids=x[1].ids()?; let mut out=Vec::new(); let mut width=None;
                for (slot,&expert) in ids.iter().enumerate() { let weight=w.get(expert as usize).ok_or_else(|| invalid("host expert ID is outside weights"))?;
                    let input=if kind==Kind::ExpertDown { let cols=*weight.shape.get(1).ok_or_else(|| invalid("host expert matrix rank"))? as usize;
                        need(f[0].len()==ids.len()*cols,"invalid host expert middle")?; Tensor::f32(&[cols as u64],f[0][slot*cols..(slot+1)*cols].to_vec())? } else { x[0].clone() };
                    let value=project(&input,weight)?;
                    need(width.is_none_or(|n| n==value.shape[0]),"expert widths differ")?; width=Some(value.shape[0]); out.extend_from_slice(value.floats()?);
                }
                one(id,&[ids.len() as u64,width.ok_or_else(|| invalid("empty host expert selection"))?],out)
            },
            Kind::ExpertSum => {
                let ids=x[0].ids()?; need(f[1].len()==ids.len() && f[2].len()==ids.len()*h,"invalid host expert sum")?;
                let mut slots=(0..ids.len()).collect::<Vec<_>>(); slots.sort_by_key(|&j| ids[j]);
                let mut out=vec![0.0;h]; for s in slots { for j in 0..h { out[j]+=f[2][s*h+j]*f[1][s]; } }
                if x.len()>3 { need(f[3].len()==h,"invalid host shared width")?; for j in 0..h { out[j]+=f[3][j]; } }
                one(id,&[h as u64],out)
            },
        }
    }
    fn kda(&self,kind: Kind,x: &[&[f32]],weights: &[&[f32]]) -> Result<Vec<(String,Tensor)>> {
        let TokenMixer::Kda(k)=self.d.layers[self.layer].mixer else { return Err(invalid("host KDA operation requires KDA")); };
        let (heads,d,taps)=(k.heads as usize,k.head_dim as usize,k.conv as usize); let width=heads*d;
        let out=|name: &str,shape: &[u64],values| -> Result<(String,Tensor)> { Ok((name.into(),Tensor::f32(shape,values)?)) };
        match kind {
            Kind::Conv => {
                need(x[..3].iter().all(|v| v.len()==width) && x[3].len()==3*(taps-1)*width
                    && weights.iter().all(|w| w.len()==width*taps),"invalid host KDA convolution")?;
                let mut result=vec![0.0;3*width]; let mut history=x[3].to_vec();
                for plane in 0..3 { for ch in 0..width { let mut sum=0.0;
                    for tap in 0..taps { let v=if tap+1==taps { x[plane][ch] } else { x[3][plane*(taps-1)*width+tap*width+ch] };
                        sum+=v*weights[plane][ch*taps+tap]; }
                    result[plane*width+ch]=sum*sigmoid(sum);
                }
                    if taps>1 { let base=plane*(taps-1)*width; history.copy_within(base+width..base+(taps-1)*width,base);
                        history[base+(taps-2)*width..base+(taps-1)*width].copy_from_slice(x[plane]); }
                }
                Ok(vec![out("convolved",&[3,width as u64],result)?,out("next_history",&[3,k.conv-1,width as u64],history)?])
            },
            Kind::KdaPrepare => {
                need(x[0].len()==3*width && x[1].len()==width && x[2].len()==heads && weights[0].len()==heads && weights[1].len()==width,"invalid host KDA prepare")?;
                let stride=4*d+1; let mut p=vec![0.0;heads*stride];
                for head in 0..heads { let start=head*d; let base=head*stride;
                    let q=&x[0][start..start+d]; let key=&x[0][width+start..width+start+d];
                    let qnorm=(dot(q,q)+1e-6).sqrt(); let knorm=(dot(key,key)+1e-6).sqrt(); let scale=1.0/(d as f32).sqrt();
                    for j in 0..d { p[base+j]=(q[j]/qnorm)*scale; p[base+d+j]=key[j]/knorm;
                        p[base+2*d+j]=x[0][2*width+start+j];
                        p[base+3*d+j]=k.gate_lower_bound as f32*sigmoid(weights[0][head].exp()*(x[1][start+j]+weights[1][start+j])); }
                    p[base+4*d]=sigmoid(x[2][head]);
                }
                Ok(vec![out("prepared",&[k.heads,stride as u64],p)?])
            },
            Kind::KdaStep => {
                let stride=4*d+1; need(x[0].len()==heads*stride && x[1].len()==heads*d*d,"invalid host KDA state")?;
                let mut state=x[1].to_vec(); let mut y=vec![0.0;width];
                for head in 0..heads { let p=&x[0][head*stride..(head+1)*stride]; let s=&mut state[head*d*d..(head+1)*d*d];
                    for key in 0..d { let decay=p[3*d+key].exp(); for value in 0..d { s[key*d+value]*=decay; } }
                    let mut delta=vec![0.0;d]; for value in 0..d { let mut prediction=0.0;
                        for key in 0..d { prediction+=s[key*d+value]*p[d+key]; }
                        delta[value]=(p[2*d+value]-prediction)*p[4*d]; }
                    for key in 0..d { for value in 0..d { s[key*d+value]+=p[d+key]*delta[value]; } }
                    for value in 0..d { let mut sum=0.0; for key in 0..d { sum+=s[key*d+value]*p[key]; } y[head*d+value]=sum; }
                }
                Ok(vec![out("recurrent",&[k.heads,k.head_dim],y)?,out("next_state",&[k.heads,k.head_dim,k.head_dim],state)?])
            },
            Kind::GatedNorm => {
                need(x[0].len()==width && x[1].len()==width && weights[0].len()==d,"invalid host KDA gated norm")?;
                let mut y=Vec::new(); for head in 0..heads { let norm=rms(&x[0][head*d..(head+1)*d],weights[0],self.d.epsilon as f32)?;
                    for j in 0..d { y.push(norm[j]*sigmoid(x[1][head*d+j])); } }
                Ok(vec![out("gated",&[width as u64],y)?])
            },
            _=>Err(invalid("unexpected host KDA operation")),
        }
    }
    fn latent(&self,id: &str,kind: Kind,x: &[Tensor],w: &[Tensor]) -> Result<Vec<(String,Tensor)>> {
        let TokenMixer::LatentAttention(a)=self.d.layers[self.layer].mixer else { return Err(invalid("host latent operation requires latent attention")); };
        let i=a.indexer; let (heads,kd,vd,lc)=(a.heads as usize,a.key_dim as usize,a.value_dim as usize,a.kv_rank as usize);
        let (ih,dim,pool)=(i.heads as usize,i.dim as usize,i.pool as usize); let ctx=self.context as usize;
        let slots=i.token_budget as usize+if i.select_tail {pool-1} else {0}; let groups=ctx/pool;
        let f=x.iter().map(|t| match &t.data { Data::F32(v)=>v.as_slice(),_=>&[] }).collect::<Vec<_>>();
        let wf=w.iter().map(Tensor::floats).collect::<Result<Vec<_>>>()?;
        let out=|name: &str,shape: &[u64],v| -> Result<(String,Tensor)> { Ok((name.into(),Tensor::f32(shape,v)?)) };
        let count=|at: usize| -> Result<usize> { let ids=x[at].ids()?; need(ids.len()==1 && ids[0] as usize<=slots,"invalid selected count")?; Ok(ids[0] as usize) };
        match kind {
            Kind::CacheAppend => { need(f[0].len()==(ctx-1)*lc && f[1].len()==lc,"invalid latent append")?;
                let mut v=f[0].to_vec(); v.extend_from_slice(f[1]); Ok(vec![out("next_latent_cache",&[ctx as u64,a.kv_rank],v)?]) },
            Kind::KeyNorm => {
                need(f[0].len()==dim && wf[0].len()==dim && wf[1].len()==dim,"invalid index key norm")?;
                let mean=fold(f[0])/dim as f32; let centered=f[0].iter().map(|&v| v-mean).collect::<Vec<_>>();
                // The saved source fixes this LayerNorm epsilon independently of RMSNorm.
                let inv=1.0/(dot(&centered,&centered)/dim as f32+INDEX_KEY_EPSILON).sqrt();
                Ok(vec![out(id,&[i.dim],(0..dim).map(|j| centered[j]*inv*wf[0][j]+wf[1][j]).collect())?])
            },
            Kind::Pool => {
                need(f[0].len()==(ctx-1)*dim && f[1].len()==f[0].len() && f[2].len()==dim && f[3].len()==dim
                    && wf[0].len()==pool*dim,"invalid index pool")?;
                let mut keys=f[0].to_vec(); keys.extend_from_slice(f[2]); let mut gates=f[1].to_vec(); gates.extend_from_slice(f[3]);
                let mut pooled=vec![0.0;groups*dim];
                for group in 0..groups { for j in 0..dim { let mut p=(0..pool).map(|s| gates[(group*pool+s)*dim+j]+wf[0][s*dim+j]).collect::<Vec<_>>();
                    softmax(&mut p); let mut value=0.0; for s in 0..pool { value+=p[s]*keys[(group*pool+s)*dim+j]; } pooled[group*dim+j]=value; } }
                Ok(vec![out("next_index_keys",&[ctx as u64,i.dim],keys)?,out("next_index_gates",&[ctx as u64,i.dim],gates)?,out("pool_keys",&[groups as u64,i.dim],pooled)?])
            },
            Kind::ScorePools => {
                need(f[0].len()==ih*dim && f[1].len()==groups*dim && f[2].len()==ih,"invalid index score")?;
                let mut scores=vec![0.0;groups]; for g in 0..groups { for head in 0..ih {
                    let score=(dot(&f[0][head*dim..(head+1)*dim],&f[1][g*dim..(g+1)*dim])*(1.0/(dim as f32).sqrt())).max(0.0);
                    scores[g]+=(f[2][head]*(1.0/(ih as f32).sqrt()))*score;
                } } Ok(vec![out(id,&[groups as u64],scores)?])
            },
            Kind::Select => {
                need(f[0].len()==groups,"invalid pool score count")?;
                let mut order=(0..groups).collect::<Vec<_>>(); sort(&mut order,f[0]); order.truncate(i.token_budget as usize/pool);
                let mut ids=order.into_iter().flat_map(|g| (g*pool..(g+1)*pool).map(|j| j as u32)).collect::<Vec<_>>();
                if i.select_tail { ids.extend((groups*pool..ctx).map(|j| j as u32)); } ids.sort_unstable();
                let used=ids.len(); ids.resize(slots,u32::MAX);
                Ok(vec![("selected".into(),Tensor { shape:vec![slots as u64],data:Data::U32(ids) }),
                    ("selected_count".into(),Tensor { shape:vec![1],data:Data::U32(vec![used as u32]) })])
            },
            Kind::Gather => {
                let used=count(2)?; let ids=x[1].ids()?; need(ids.len()==slots && f[0].len()==ctx*lc,"invalid latent gather shape")?;
                let mut gathered=vec![0.0;slots*lc]; let mut previous=None;
                for (s,&j) in ids[..used].iter().enumerate() { need((j as usize)<ctx && previous.is_none_or(|p| p<j),"invalid causal latent selection")?;
                    gathered[s*lc..(s+1)*lc].copy_from_slice(&f[0][j as usize*lc..(j as usize+1)*lc]); previous=Some(j); }
                Ok(vec![out("selected_latent",&[slots as u64,a.kv_rank],gathered)?])
            },
            Kind::ExpandKv => {
                let used=count(1)?; need(f[0].len()==slots*lc,"invalid gathered latent extent")?;
                let mut keys=vec![0.0;heads*slots*kd]; let mut values=vec![0.0;heads*slots*vd];
                for s in 0..used { let v=project(&Tensor::f32(&[a.kv_rank],f[0][s*lc..(s+1)*lc].to_vec())?,&w[0])?;
                    need(v.shape==[a.heads*(a.key_dim+a.value_dim)],"invalid KV up rows")?;
                    let v=v.floats()?; for head in 0..heads { let base=head*(kd+vd);
                        keys[(s*heads+head)*kd..(s*heads+head+1)*kd].copy_from_slice(&v[base..base+kd]);
                        values[(s*heads+head)*vd..(s*heads+head+1)*vd].copy_from_slice(&v[base+kd..base+kd+vd]); }
                }
                Ok(vec![out("keys",&[slots as u64,a.heads,a.key_dim],keys)?,out("values",&[slots as u64,a.heads,a.value_dim],values)?])
            },
            Kind::Scores => {
                let used=count(2)?; need(f[0].len()==heads*kd && f[1].len()==heads*slots*kd,"invalid attention score shape")?;
                check_local_slots(&x[3],slots,used)?;
                let mut scores=vec![0.0;heads*slots]; for head in 0..heads { for s in 0..used {
                    scores[head*slots+s]=dot(&f[0][head*kd..(head+1)*kd],&f[1][(s*heads+head)*kd..(s*heads+head+1)*kd])*(1.0/(kd as f32).sqrt());
                } } Ok(vec![out(id,&[a.heads,slots as u64],scores)?])
            },
            Kind::Softmax => {
                let used=count(1)?; need(f[0].len()==heads*slots,"invalid attention softmax shape")?;
                let mut p=vec![0.0;heads*slots]; for head in 0..heads { if used>0 { p[head*slots..head*slots+used].copy_from_slice(&f[0][head*slots..head*slots+used]); softmax(&mut p[head*slots..head*slots+used]); } }
                Ok(vec![out("probabilities",&[a.heads,slots as u64],p)?])
            },
            Kind::Values => {
                let used=count(2)?; need(f[0].len()==heads*slots && f[1].len()==heads*slots*vd,"invalid attention value shape")?;
                check_local_slots(&x[3],slots,used)?;
                let mut values=vec![0.0;heads*vd]; for head in 0..heads { for j in 0..vd { for s in 0..used {
                    values[head*vd+j]+=f[0][head*slots+s]*f[1][(s*heads+head)*vd+j];
                } } } Ok(vec![out("attended",&[a.heads*a.value_dim],values)?])
            },
            _=>Err(invalid("unexpected host latent operation")),
        }
    }
}
pub(crate) fn project(x: &Tensor,w: &Tensor) -> Result<Tensor> {
    let input=x.floats()?; let weight=w.floats()?;
    need(w.shape.len()==2 && w.shape[1] as usize==input.len() && !input.is_empty(),"invalid host projection shape")?;
    Tensor::f32(&[w.shape[0]],weight.chunks_exact(input.len()).map(|row| dot(row,input)).collect())
}
fn check_local_slots(t: &Tensor,slots: usize,used: usize) -> Result<()> {
    let ids=t.ids()?; need(ids.len()==slots && ids[..used].iter().enumerate().all(|(j,&v)| v==j as u32),"invalid local attention slots")
}
fn dot(a: &[f32],b: &[f32]) -> f32 { let mut sum=0.0; for j in 0..a.len() { sum+=a[j]*b[j]; } sum }
fn fold(a: &[f32]) -> f32 { a.iter().fold(0.0,|sum,&v| sum+v) }
fn sigmoid(v: f32) -> f32 { 1.0/(1.0+(-v).exp()) }
pub(crate) fn rms(x: &[f32],w: &[f32],eps: f32) -> Result<Vec<f32>> {
    need(!x.is_empty() && x.len()==w.len(),"invalid host RMS norm")?;
    let inv=1.0/(dot(x,x)/x.len() as f32+eps).sqrt(); Ok(x.iter().zip(w).map(|(&v,&w)| (v*inv)*w).collect())
}
fn softmax(x: &mut [f32]) {
    let max=x.iter().copied().fold(f32::NEG_INFINITY,f32::max); for v in x.iter_mut() { *v=(*v-max).exp(); }
    let sum=fold(x); for v in x { *v/=sum; }
}
fn sort(ids: &mut [usize],scores: &[f32]) { ids.sort_by(|&a,&b| if scores[a]==scores[b] { a.cmp(&b) } else { scores[b].total_cmp(&scores[a]) }); }
