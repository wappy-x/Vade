//! Bounded causal recipe interpreter. Full service admission stays separate.
use crate::{Result,causal_layer,mxfp8_projection::Mxfp8Projection,layer_driver::LayerBoundary};
use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan,LayerSchedule,Topology},state::StatePlan,storage::WeightStorage,description::TokenMixer},safetensors::Dtype,flash::{nvfp4,activation_nvfp4::Activation}};
use spark_gpu_probe::{causal_ops::{Shape,Op},norm_boundary::{NormShape,WeightType}};
use std::{cell::RefCell,collections::BTreeMap};
pub const PROMPT_ROWS:usize=4;
fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn shape(op:Op)->Shape {Shape {op,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}}
fn number(n:u64)->Result<u32> {n.try_into().map_err(|_|"bounded recipe dimension exceeds ABI".into())}
fn plain(c:&CheckpointAdmission,name:&str)->Result<Vec<f32>> {
    let u=c.upload(name).map_err(|e|e.to_string())?;need(u.storage==WeightStorage::Plain,"parameter requires a plain recipe")?;
    let v=match u.dtype {Dtype::Bf16=>u.weight.chunks_exact(2).map(|b|f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap()))<<16)).collect::<Vec<_>>(),
        Dtype::F32=>u.weight.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect(),_=>return Err("plain parameter precision is unsupported".into())};
    need(v.iter().all(|v|v.is_finite()),"plain parameter is not finite")?;Ok(v)
}
fn norm_shape(c:&CheckpointAdmission,name:&str,width:usize)->Result<NormShape> {
    let w=c.headers().weights().get(name).ok_or("missing norm weight")?;
    let dtype=match w.components()[0].dtype {Dtype::Bf16=>WeightType::Bf16,Dtype::F32=>WeightType::F32,_=>return Err("unsupported norm precision".into())};
    NormShape::new(width,c.description().epsilon as f32,dtype)
}
#[cfg(any(target_os="linux",test))]
mod lineage;
mod residual_plan;
pub trait Compute {
    /// One ordered execution owns temporary buffers through state publication.
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {run()}
    fn resident_weights_ready(&self)->bool {false}
    fn fatal_error(&self)->Option<String> {None}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {None}
    fn serving_trace(&self)->bool {false}
    fn profile_tick_begin(&self,_tick:u64) {}
    fn profile_tick_end(&self) {}
    fn plan_cost(&self,_ns:u128) {}
    fn trace_start(&self) {}
    fn cost_report(&self,_phase:&str) {}
    type Value;type Pass;
    fn projects_complete_rows(&self)->bool {false}
    fn prepare(&self,_:&CheckpointAdmission)->Result<()> {Ok(())}
    fn projection(&self,_:&CheckpointAdmission,_:&str,_:&Self::Value,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    fn projection_rows(&self,_:&CheckpointAdmission,_:&str,_:&[&Self::Value],_:bool)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn projection_batch(&self,_:&CheckpointAdmission,_:&[(&str,&Self::Value)],_:bool)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn repeated_projection(&self,c:&CheckpointAdmission,name:&str,x:&Self::Value,round:bool)->Result<Option<Self::Value>> {self.projection(c,name,x,round)}
    fn attention(&self,_heads:u32,_key_dim:u32,_value_dim:u32,_slots:usize,_capacity:usize,_x:[&Self::Value;4])->Result<Option<Self::Value>> {Ok(None)}
    fn embedding(&self,_:u32,_:usize)->Result<Option<Self::Value>> {Ok(None)}
    fn upload(&self,v:&[f32])->Result<Self::Value>;
    fn read(&self,v:&Self::Value)->Result<Vec<f32>>;
    fn range(&self,v:&Self::Value,start:usize,len:usize)->Result<Self::Value>;
    fn join(&self,parts:&[&Self::Value])->Result<Self::Value>;
    fn zeros(&self,len:usize)->Result<Self::Value>;
    fn operation(&self,s:Shape,x:[Option<&Self::Value>;4],w:&[f32])->Result<Self::Value>;
    fn norm(&self,name:&str,s:NormShape,x:&Self::Value,w:&[f32])->Result<Self::Value>;
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value)->Result<(Self::Pass,Self::Value)>;
    fn finish(&self,p:Self::Pass,x:&Self::Value)->Result<Self::Value>;
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Self::Value)->Result<Self::Value>;
    fn round(&self,x:&Self::Value)->Result<Self::Value>;
    fn mean(&self,x:&Self::Value,streams:u32,hidden:u32)->Result<Self::Value>;
    /// One operation over token-major prompt rows.
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Self::Value>;5],w:&[f32])->Result<Self::Value>;
    fn rows_embedding(&self,_:&[u32],_:usize)->Result<Option<Self::Value>> {Ok(None)}
    /// Token-major `[m,rows]`, complete on every rank. None runs one row at a time.
    fn rows_project(&self,_:&CheckpointAdmission,_:&str,_:&Self::Value,_:usize,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    /// One layer's selected experts in one pass. `first` and `second` list
    /// (matrix, token rows) over contiguous `source` rows. With `second`, the
    /// two local outputs meet in SwiGLU before the gather. `places` has three
    /// numbers for each output row. None runs each expert separately.
    fn rows_experts(&self,_:&[(&str,usize)],_:Option<(&[(&str,usize)],f32)>,_:&Self::Value,_:&[f32])->Result<Option<Self::Value>> {Ok(None)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&Self::Value,m:usize,w:&[f32])->Result<Self::Value>;
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value,m:usize)->Result<(Self::Pass,Self::Value)>;
    fn rows_finish(&self,p:Self::Pass,x:&Self::Value,m:usize)->Result<Self::Value>;
}
/// `expanded` keeps each prompt layer's expanded latent rows. Only prompt groups
/// use it, and it holds only pure projections of `latent_cache` rows.
pub struct State<V> {pub position:u64,pub layers:Vec<BTreeMap<String,V>>,pub expanded:BTreeMap<usize,(std::rc::Rc<V>,usize)>}
pub struct Decoder<'f,C:Compute> {checkpoint:&'f CheckpointAdmission,compute:C,capacity:u64,parameters:BTreeMap<String,Vec<f32>>,
    blocks:BTreeMap<(usize,usize),crate::layer_operands::ResidualBlockOperands>,schedules:RefCell<Vec<LayerSchedule>>,expanded_limit:std::cell::Cell<usize>}
impl<'f,C:Compute> Decoder<'f,C> {
    pub fn new(checkpoint:&'f CheckpointAdmission,compute:C,capacity:u64)->Result<Self> {
        let d=checkpoint.description();need(!d.layers.is_empty() && d.hidden>0 && d.residual.streams>0 && d.vocab>0 && d.vocab<=u32::MAX as u64 && !d.tied_embeddings,"unsupported bounded text geometry")?;
        need(capacity>0 && capacity<=checkpoint.description().max_positions,"bounded decoder capacity exceeds the recipe")?;
        let mut parameters=BTreeMap::new();let mut blocks=BTreeMap::new();let mut schedules=Vec::new();let topology=Topology {rank:0,parts:1};
        let storage=StatePlan::new(d,topology,capacity,isize::MAX as usize).map_err(|e|e.to_string())?;
        for layer in 0..d.layers.len() {let boundary=LayerBoundary::new(d,checkpoint.headers(),&storage,layer,1,topology,usize::MAX)?;
            residual_plan::check(boundary.program())?;
            schedules.push(LayerSchedule::new(d,layer,topology).map_err(|e|e.to_string())?);
            for (index,step) in boundary.program().steps.iter().enumerate() {
                if step.kind==Kind::MhcPrepare {blocks.insert((layer,index),boundary.prepare_residual_block_checkpoint(index,checkpoint)?);}
                for w in &step.weights {if w.storage==WeightStorage::Plain && !parameters.contains_key(&w.name) {parameters.insert(w.name.clone(),plain(checkpoint,&w.name)?);}}
            }
        }
        let name="model.language_model.norm.weight";parameters.insert(name.into(),plain(checkpoint,name)?);
        compute.prepare(checkpoint)?;Ok(Self {checkpoint,compute,capacity,parameters,blocks,schedules:RefCell::new(schedules),expanded_limit:std::cell::Cell::new(rows::EXPANDED_ROWS)})
    }
    pub fn zero_state(&self)->Result<State<C::Value>> {
        let mut layers=Vec::new();for layer in 0..self.checkpoint.description().layers.len() {
            let p=LayerPlan::new(self.checkpoint.description(),layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;let mut inputs=BTreeMap::new();
            for b in p.inputs.iter().filter(|b|b.name!="residual" && b.name!="local_slots") {inputs.insert(b.name.clone(),self.compute.zeros(b.elements().map_err(|e|e.to_string())?)?);}layers.push(inputs);
        }Ok(State {position:0,layers,expanded:BTreeMap::new()})
    }
    /// No committed state changes until every decoder layer and the head succeed.
    pub fn step(&self,token:u32,state:&State<C::Value>,mut observe:impl FnMut(usize,&C::Value)->Result<()>)->Result<(Vec<f32>,u32,State<C::Value>)> {
        let c=self.checkpoint;let d=c.description();need(state.position<self.capacity && state.layers.len()==d.layers.len(),"bounded state context differs")?;
        need(u64::from(token)<d.vocab,"bounded input token exceeds vocabulary")?;
        let mut residual=self.embedding(token)?;let mut layers=Vec::new();
        for layer in 0..d.layers.len() {let (mut values,next)=self.layer_group(layer,state.position+1,&[&residual],&state.layers[layer],state.expanded.get(&layer))?;let output=values.remove(0);observe(layer,&output)?;residual=output;layers.push(next);}
        let h=d.hidden as usize;let mean=self.compute.mean(&residual,number(d.residual.streams)?,number(d.hidden)?)?;
        let norm_name="model.language_model.norm.weight";let normalized=self.compute.norm(norm_name,norm_shape(c,norm_name,h)?,&mean,&self.parameters[norm_name])?;
        let head=Mxfp8Projection::matrix(c,"lm_head.weight",WeightType::F32)?;let logits=self.compute.read(&self.compute.mx(&head,&normalized)?)?;
        need(logits.len()==d.vocab as usize && logits.iter().all(|v|v.is_finite()),"bounded logits extent or value differs")?;
        let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
        Ok((logits,best as u32,State {position:state.position+1,layers,expanded:state.expanded.clone()}))
    }
    /// Layer-major prompt rows. State producers run in ascending token order;
    /// each matrix row retains the scalar feature fold and rounding boundaries.
    pub fn prefill(&self,tokens:&[u32],state:&State<C::Value>)->Result<(Vec<Vec<f32>>,State<C::Value>)> {
        need(!tokens.is_empty() && tokens.len()<=PROMPT_ROWS && state.layers.len()==self.checkpoint.description().layers.len(),"bounded prompt group differs")?;
        need(state.position.checked_add(tokens.len() as u64).is_some_and(|n|n<=self.capacity),"bounded prompt exceeds capacity")?;
        let mut residuals=tokens.iter().map(|&token|self.embedding(token)).collect::<Result<Vec<_>>>()?;let mut layers=Vec::new();
        for layer in 0..self.checkpoint.description().layers.len() {
            let (next,state)=self.layer_group(layer,state.position+1,&residuals.iter().collect::<Vec<_>>(),&state.layers[layer],None)?;
            residuals=next;layers.push(state);
        }
        let c=self.checkpoint;let d=c.description();let name="model.language_model.norm.weight";let mut logits=Vec::new();
        for residual in residuals {
            let mean=self.compute.mean(&residual,number(d.residual.streams)?,number(d.hidden)?)?;
            let norm=self.compute.norm(name,norm_shape(c,name,d.hidden as usize)?,&mean,&self.parameters[name])?;
            let head=Mxfp8Projection::matrix(c,"lm_head.weight",WeightType::F32)?;
            let row=self.compute.read(&self.compute.mx(&head,&norm)?)?;
            need(row.len()==d.vocab as usize && row.iter().all(|v|v.is_finite()),"bounded prompt logits differ")?;logits.push(row);
        }Ok((logits,State {position:state.position+tokens.len() as u64,layers,expanded:BTreeMap::new()}))
    }
    fn embedding(&self,token:u32)->Result<C::Value> {
        let c=self.checkpoint;let d=c.description();need(u64::from(token)<d.vocab,"bounded embedding token exceeds vocabulary")?;
        if let Some(value)=self.compute.embedding(token,d.residual.streams as usize)? {return Ok(value);}
        let embedding="model.language_model.embed_tokens.weight";let file=c.component_file(embedding).map_err(|e|e.to_string())?;
        let mut bytes=vec![0;d.hidden as usize*2];file.read_range(embedding,u64::from(token)*d.hidden*2,&mut bytes).map_err(|e|e.to_string())?;
        let row=spark_gpu_probe::text_reference::embedding(&bytes)?;self.compute.upload(&row.repeat(d.residual.streams as usize))
    }
    fn project(&self,name:&str,input:&C::Value,round:bool)->Result<C::Value> {
        if let Some(output)=self.compute.projection(self.checkpoint,name,input,round)? {return Ok(output);}
        let c=self.checkpoint;let view=c.headers().weights().get(name).ok_or("missing bounded matrix")?;let logical=view.logical();
        if logical.storage==WeightStorage::Mxfp8 {return self.compute.mx(&Mxfp8Projection::matrix(c,name,WeightType::F32)?,input);}
        let u=c.upload(name).map_err(|e|e.to_string())?;need(u.shape.len()==2,"bounded projection requires a matrix")?;
        let (rows,cols)=(u.shape[0] as usize,u.shape[1] as usize);let mut s=shape(Op::Project);s.rows=number(u.shape[0])?;s.cols=number(u.shape[1])?;
        let value=match u.storage {
            WeightStorage::Plain=>self.compute.operation(s,[Some(input),None,None,None],&plain(c,name)?)?,
            WeightStorage::Fp8Block {block}=>{
                let scales=u.scales.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                let (br,bc)=(block[0] as usize,block[1] as usize);let grid=cols.div_ceil(bc);
                let weight=u.weight.iter().enumerate().map(|(i,&v)|nvfp4::e4m3(v)*scales[(i/cols/br)*grid+i%cols/bc]).collect::<Vec<_>>();
                self.compute.operation(s,[Some(&self.compute.round(input)?),None,None,None],&weight)?
            },
            WeightStorage::Nvfp4 {..}=>{
                let activation=Activation::from_bf16_source(&self.compute.read(input)?,u.input_scale.ok_or("NVFP4 calibrated input scale is absent")?).map_err(|e|e.to_string())?;
                let combined=u.global_scale.ok_or("NVFP4 weight global scale is absent")?*activation.global();
                let a=(0..cols).map(|i|nvfp4::e2m1(activation.codes()[i/2]>>((i%2)*4))).collect::<Vec<_>>();
                let weight=(0..rows*cols).map(|i|nvfp4::e2m1(u.weight[i/2]>>((i%2)*4))).collect::<Vec<_>>();
                let scales=u.scales.iter().enumerate().map(|(i,&v)|(nvfp4::e4m3(v)*nvfp4::e4m3(activation.scales()[i%(cols/16)]))*combined).collect::<Vec<_>>();
                s.op=Op::Nvfp4;self.compute.operation(s,[Some(&self.compute.upload(&a)?),Some(&self.compute.upload(&scales)?),None,None],&weight)?
            },
            _=>return Err("bounded projection storage is unsupported".into()),
        };
        if round {self.compute.round(&value)} else {Ok(value)}
    }
    pub fn layer(&self,layer:usize,context:u64,residual:&C::Value,state:&BTreeMap<String,C::Value>)->Result<(C::Value,BTreeMap<String,C::Value>)> {
        let (mut values,state)=self.layer_group(layer,context,&[residual],state,None)?;
        Ok((values.remove(0),state))
    }
    fn layer_group(&self,layer:usize,first_context:u64,residuals:&[&C::Value],state:&BTreeMap<String,C::Value>,expanded:Option<&(std::rc::Rc<C::Value>,usize)>)->Result<(Vec<C::Value>,BTreeMap<String,C::Value>)> {
        need(!residuals.is_empty() && residuals.len()<=PROMPT_ROWS,"bounded prompt row count differs")?;
        let count=residuals.len();let c=self.checkpoint;let d=c.description();let b=&self.compute;
        need(first_context>0 && first_context.checked_add(count as u64-1).is_some_and(|n|n<=self.capacity && n<=u32::MAX as u64),"bounded layer context exceeds capacity")?;
        let planning=std::time::Instant::now();
        let mut schedules=self.schedules.borrow_mut();let schedule=schedules.get_mut(layer).ok_or("bounded layer is outside recipe")?;
        let plans=(0..count).map(|row|schedule.at(first_context+row as u64).cloned().map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;drop(schedules);
        b.plan_cost(planning.elapsed().as_nanos());
        let names=match d.layers[layer].mixer {
            TokenMixer::Kda(_)=>vec![("kda.history","next_history"),("kda.state","next_state")],
            TokenMixer::LatentAttention(_)=>vec![("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates")],
        };
        let mut all_values=(0..count).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        for (values,residual) in all_values.iter_mut().zip(residuals) {values.insert("residual".to_owned(),b.range(residual,0,d.hidden as usize*d.residual.streams as usize)?);}
        let mut all_passes=(0..count).map(|_|BTreeMap::new()).collect::<Vec<_>>();let mut skip=0;let mut projected=std::collections::BTreeSet::new();
        for index in 0..plans[0].steps.len() {
            if skip>0 {skip-=1;continue;}
            if projected.remove(&index) {continue;}
            let first=&plans[0].steps[index];
            need(plans.iter().all(|p|p.steps.len()==plans[0].steps.len() && p.steps[index].kind==first.kind && p.steps[index].id==first.id),"prompt row programs differ")?;
            if first.kind==Kind::Project {
                for row in 0..count {self.import_row_state(row,index,&plans,&names,state,&mut all_values)?;}
                if count==1 {let done=self.independent_batch(index,&plans[0],&mut all_values[0])?;
                    if !done.is_empty() {projected.extend(done);continue;}
                }
                let inputs=(0..count).map(|row|all_values[row].get(&plans[row].steps[index].inputs[0]).ok_or_else(||"prompt projection input is absent".into())).collect::<Result<Vec<_>>>()?;
                let router=plans[0].steps.iter().any(|s|s.kind==Kind::Route && s.inputs.contains(&first.outputs[0].name));
                let output=self.project_rows(&first.weights[0].name,&inputs,!router)?;
                need(output.len()==count,"prompt projection row count differs")?;
                for (row,value) in output.into_iter().enumerate() {all_values[row].insert(plans[row].steps[index].outputs[0].name.clone(),value);}continue;
            }
            if matches!(first.kind,Kind::ExpertProject|Kind::ExpertDown) {
                if count==1 {
                    let span=if first.kind==Kind::ExpertProject && plans[0].steps.get(index+1).is_some_and(|next|next.kind==Kind::ExpertProject && next.inputs==first.inputs) {2} else {1};
                    if self.expert_batch(index,span,&plans[0],&mut all_values[0])? {skip=span-1;continue;}
                }
                self.expert_rows(index,&plans,&mut all_values)?;continue;
            }
            for row in 0..count {
                self.import_row_state(row,index,&plans,&names,state,&mut all_values)?;
                let plan=&plans[row];let step=&plan.steps[index];let context=first_context+row as u64;
                let values=&mut all_values[row];let passes=&mut all_passes[row];
                if step.kind==Kind::MhcPrepare {
                    let key=(layer,index);let p=self.blocks.get(&key).ok_or("resident residual parameters are absent")?;let (pass,norm)=b.begin(key,p,&values[&step.inputs[0]])?;
                    passes.insert(step.id.split('.').next().unwrap(),pass);values.insert(plan.steps[index+2].outputs[0].name.clone(),norm);skip=2;continue;
                }
                if step.kind==Kind::Combine {let side=step.id.split('.').next().unwrap();let pass=passes.remove(side).ok_or("bounded residual pass is absent")?;
                    let value=b.finish(pass,&values[&step.inputs[2]])?;values.insert(step.outputs[0].name.clone(),value);continue;}
            if count==1 && step.kind==Kind::Scores {
                if let (Some(softmax),Some(value))=(plan.steps.get(index+1),plan.steps.get(index+2)) {
                    let private=plan.returns.iter().all(|name|name!=&step.outputs[0].name && name!=&softmax.outputs[0].name) && plan.steps[index+3..].iter().all(|later|later.inputs.iter().all(|name|name!=&step.outputs[0].name && name!=&softmax.outputs[0].name));
                    if softmax.kind==Kind::Softmax && value.kind==Kind::Values && private && softmax.inputs[0]==step.outputs[0].name && value.inputs[0]==softmax.outputs[0].name && softmax.inputs[1]==step.inputs[2] && value.inputs[2]==step.inputs[2] {
                        if let TokenMixer::LatentAttention(a)=d.layers[layer].mixer {
                            let slots=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}).min(context) as usize;
                            let capacity=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}).min(self.capacity) as usize;
                            let args=[&values[&step.inputs[0]],&values[&step.inputs[1]],&values[&value.inputs[1]],&values[&step.inputs[2]]];
                            if let Some(output)=b.attention(number(a.heads)?,number(a.key_dim)?,number(a.value_dim)?,slots,capacity,args)? {
                                values.insert(value.outputs[0].name.clone(),output);skip=2;continue;
                            }
                        }
                    }
                }
            }
            let input=step.inputs.iter().filter(|n|n.as_str()!="local_slots").map(|n|values.get(n).ok_or_else(||format!("missing bounded input {n}"))).collect::<Result<Vec<_>>>()?;
            let output:Vec<C::Value>=match step.kind {
                Kind::Norm=>{let name=&step.weights[0].name;let width=step.outputs[0].elements().map_err(|e|e.to_string())?;
                    vec![b.norm(name,norm_shape(c,name,width)?,input[0],&self.parameters[name])?]},
                Kind::CacheAppend=>vec![b.join(&[input[0],input[1]])?],
                Kind::ExpandKv=>{
                    let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("KV expansion requires latent recipe".into());};
                    let count=b.read(input[1])?[0] as usize;let slots=step.outputs[0].shape[0].min(context) as usize;
                    need(count<=slots,"bounded KV count exceeds slots")?;let (h,kd,vd,lc)=(a.heads as usize,a.key_dim as usize,a.value_dim as usize,a.kv_rank as usize);
                    if slots==0 {vec![b.zeros(0)?,b.zeros(0)?]} else {
                    let mut projected=Vec::new();
                    // Prompt groups already hold pure expansions of their rows. A run of
                    // consecutive selected rows needs one copy and no latent lookup.
                    let row=h*(kd+vd);let ids=if expanded.is_some() {b.read(values.get("selected").ok_or("bounded selection is absent")?)?} else {Vec::new()};
                    let mut slot=0;while slot<count {
                        if let Some((rows,held))=expanded {let id=ids[slot] as usize;if id<*held {
                            let mut end=slot+1;while end<count && ids[end] as usize==id+(end-slot) && (ids[end] as usize)<*held {end+=1;}
                            projected.push(b.range(rows,id*row,(end-slot)*row)?);slot=end;continue;}}
                        let latent=b.range(input[0],slot*lc,lc)?;let name=&step.weights[0].name;
                        let value=if let Some(value)=b.repeated_projection(c,name,&latent,true)? {value} else {self.project(name,&latent,true)?};
                        projected.push(value);slot+=1;}
                    let packed=b.join(&projected.iter().collect::<Vec<_>>())?;
                    let mut s=shape(Op::UnpackKv);s.rows=number(slots as u64)?;s.heads=number(h as u64)?;s.dim=number(kd as u64)?;
                    s.cols=number(vd as u64)?;s.taps=number(count as u64)?;
                    let unpacked=b.operation(s,[Some(&packed),None,None,None],&[])?;let keys=slots*h*kd;
                    vec![b.range(&unpacked,0,keys)?,b.range(&unpacked,keys,slots*h*vd)?]}
                },
                kind=>{
                    let mut s=shape(match kind {Kind::Conv=>Op::Conv,Kind::KdaPrepare=>Op::Prepare,Kind::KdaStep=>Op::Step,Kind::GatedNorm=>Op::GatedNorm,
                        Kind::SwiGlu=>Op::SwiGlu,Kind::KeyNorm=>Op::KeyNorm,Kind::Pool=>Op::Pool,Kind::ScorePools=>Op::PoolScores,Kind::Select=>Op::Select,
                        Kind::Gather=>Op::Gather,Kind::Scores=>Op::Scores,Kind::Softmax=>Op::Softmax,Kind::Values=>Op::Values,Kind::Route=>Op::Route,Kind::ExpertSum=>Op::ExpertFold,
                        _=>return Err(format!("{}: bounded operation is unsupported",step.id))});
                    s.epsilon=d.epsilon as f32;
                    match d.layers[layer].mixer {
                        TokenMixer::Kda(k)=>{s.heads=number(k.heads)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;if kind==Kind::KdaPrepare {s.bound=k.gate_lower_bound as f32;}},
                        TokenMixer::LatentAttention(a)=>{
                            s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;
                            match kind {
                                Kind::KeyNorm=>s.epsilon=plan.settings.index_key_epsilon,
                                Kind::Pool=>s.rows=number(context)?,
                                Kind::ScorePools=>s.rows=number(context/a.indexer.pool)?,
                                Kind::Select=>{s.rows=number(context)?;s.taps=number(a.indexer.token_budget)?;s.dim=number(a.indexer.pool)?;s.cols=u32::from(a.indexer.select_tail);},
                                Kind::Gather=>{s.rows=number(context)?;s.dim=number(a.kv_rank)?;s.cols=number(step.outputs[0].shape[0].min(context))?;},
                                Kind::Scores|Kind::Softmax|Kind::Values=>{s.heads=number(a.heads)?;s.dim=number(if kind==Kind::Values {a.value_dim} else {a.key_dim})?;
                                    s.rows=number((a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}).min(context))?;},
                                _=>{},
                            }
                        },
                    }
                    if kind==Kind::SwiGlu {s.rows=step.outputs[0].elements().map_err(|e|e.to_string())?.try_into().map_err(|_|"bounded middle exceeds ABI")?;s.bound=d.swiglu_limit as f32;}
                    if kind==Kind::Route {s.heads=number(d.routing.groups)?;s.dim=number(d.routing.selected_groups)?;s.rows=number(step.weights[0].shape[0])?;
                        s.cols=number(step.outputs[0].shape[0])?;s.taps=u32::from(d.routing.normalize_selected);s.bound=d.routing.scale as f32;}
                    if kind==Kind::ExpertSum {s.rows=number(plan.steps.iter().find(|s|s.kind==Kind::Route).ok_or("bounded expert fold has no route")?.outputs[0].shape[0])?;
                        s.cols=number(d.hidden)?;s.taps=u32::from(input.len()==4);}
                    let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                    let packed;if kind==Kind::Conv {packed=Some(b.join(&input[..3])?);} else {packed=None;}
                    // Select retains its complete diagnostic row. Gather reads only the
                    // causal prefix; padded slots never enter a live reduction.
                    let selected=if kind==Kind::Gather {Some(b.range(input[1],0,s.cols as usize)?)} else {None};
                    let args=if let Some(v)=selected.as_ref() {[Some(input[0]),Some(v),Some(input[2]),None]}
                        else if let Some(v)=packed.as_ref() {[Some(v),Some(input[3]),None,None]} else {std::array::from_fn(|i|input.get(i).copied())};
                    // The same total order and output layout without the repeated scans.
                    let result=if kind==Kind::Select {
                        let groups=context/s.dim as u64;let mut row=spark_gpu_probe::prompt_ops::Shape::new(spark_gpu_probe::prompt_ops::Op::SelectRows,1).ok_or("selection row count")?;
                        row.dim=s.dim;row.taps=s.taps;row.rows=number(groups)?;row.cols=s.cols;row.first=number(context)?;
                        let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("selection requires a latent recipe".into());};
                        row.slots=number(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0})?;
                        let out=b.prompt(row,[args[0],None,None,None,None],&[])?;
                        need(b.read(&out)?[row.slots as usize]>=0.0,"selection score is not a number")?;out
                    } else {b.operation(s,args,&weight)?};
                    let (_,_,units,words)=s.lengths().ok_or("invalid bounded operation shape")?;
                    match kind {
                        Kind::Conv|Kind::KdaStep=>{
                            let first=if kind==Kind::Conv {s.heads as usize*s.dim as usize} else {s.dim as usize*s.dim as usize};
                            let mut a=Vec::new();let mut c=Vec::new();for i in 0..units {a.push(b.range(&result,i*words,first)?);c.push(b.range(&result,i*words+first,words-first)?);}
                            let a=b.join(&a.iter().collect::<Vec<_>>())?;let c=b.join(&c.iter().collect::<Vec<_>>())?;
                            if kind==Kind::KdaStep {vec![c,a]} else {vec![a,c]}
                        },
                        Kind::Pool=>{let len=context as usize*s.dim as usize;vec![b.range(&result,0,len)?,b.range(&result,len,len)?,b.range(&result,2*len,(context as usize/s.taps as usize)*s.dim as usize)?]},
                        Kind::Select=>{let slots=words-1;vec![b.range(&result,0,slots)?,b.range(&result,slots,1)?]},
                        Kind::Route=>vec![b.range(&result,0,s.cols as usize)?,b.range(&result,s.cols as usize,s.cols as usize)?],
                        Kind::ScorePools if s.rows==0=>vec![b.zeros(0)?],
                        _=>vec![result],
                    }
                },
            };
            need(output.len()==step.outputs.len(),"bounded output count differs")?;
            for (spec,v) in step.outputs.iter().zip(output) {values.insert(spec.name.clone(),v);}
            }
        }
        let mut output=Vec::new();for values in &mut all_values {output.push(values.remove("output").ok_or("bounded layer has no output")?);}
        let last=all_values.last_mut().ok_or("empty prompt layer")?;let mut next=BTreeMap::new();
        for (input,output) in names {next.insert(input.into(),last.remove(output).ok_or("bounded layer has no next state")?);}Ok((output,next))
    }
    fn import_row_state(&self,row:usize,index:usize,plans:&[LayerPlan],names:&[(&str,&str)],state:&BTreeMap<String,C::Value>,values:&mut [BTreeMap<String,C::Value>])->Result<()> {
        for input in &plans[row].steps[index].inputs {
            if values[row].contains_key(input) || input=="local_slots" {continue;}
            let Some(spec)=plans[row].inputs.iter().find(|spec|&spec.name==input) else {continue;};
            let source=if row==0 {state.get(input).ok_or("bounded layer state input is absent")?} else {
                let output=names.iter().find(|(name,_)|*name==input).ok_or("prompt state output binding is absent")?.1;
                values[row-1].get(output).ok_or("prompt state predecessor has not completed")?
            };
            let value=self.compute.range(source,0,spec.elements().map_err(|e|e.to_string())?)?;values[row].insert(input.clone(),value);
        }Ok(())
    }
    fn project_rows(&self,name:&str,input:&[&C::Value],round:bool)->Result<Vec<C::Value>> {
        if let Some(values)=self.compute.projection_rows(self.checkpoint,name,input,round)? {return Ok(values);}
        input.iter().map(|row|self.project(name,row,round)).collect()
    }
    fn expert_batch(&self,index:usize,span:usize,plan:&LayerPlan,values:&mut BTreeMap<String,C::Value>)->Result<bool> {
        let compute=&self.compute;let first=&plan.steps[index];let ids=compute.read(&values[&first.inputs[1]])?;
        let mut selected=Vec::new();
        for (slot,id) in ids.into_iter().enumerate() {
            need(id>=0.0 && id.fract()==0.0 && (id as usize)<first.weights.len(),"bounded batch expert ID is outside recipe")?;
            let id=id as usize;need(!selected.iter().any(|&(_,prior)|prior==id),"bounded batch route repeats an expert")?;selected.push((slot,id));
        }
        selected.sort_by_key(|&(_,id)|id);
        let mut owned=Vec::new();let mut names=Vec::new();let mut slots=Vec::new();
        for stage in index..index+span {let step=&plan.steps[stage];
            for &(slot,id) in &selected {
                let weight=step.weights.get(id).ok_or("bounded batch expert weight is absent")?;
                let source=values.get(&step.inputs[0]).ok_or("bounded batch input is absent")?;
                owned.push(if step.kind==Kind::ExpertDown {let width=usize::try_from(weight.shape[1]).map_err(|_|"bounded batch feature extent overflow")?;
                    Some(compute.range(source,slot.checked_mul(width).ok_or("bounded batch slot overflow")?,width)?)} else {None});
                names.push(weight.name.as_str());slots.push((stage-index,slot));
            }
        }
        let input=names.iter().enumerate().map(|(job,name)|(*name,owned[job].as_ref().unwrap_or(&values[&plan.steps[index+slots[job].0].inputs[0]]))).collect::<Vec<_>>();
        let Some(projected)=compute.projection_batch(self.checkpoint,&input,true)? else {return Ok(false);};
        need(projected.len()==slots.len(),"bounded batch output count differs")?;
        let mut outputs=(0..span).map(|_|(0..selected.len()).map(|_|None).collect::<Vec<_>>()).collect::<Vec<_>>();
        for ((stage,slot),value) in slots.into_iter().zip(projected) {outputs[stage][slot]=Some(value);}
        for (stage,output) in outputs.into_iter().enumerate() {
            let refs=output.iter().map(|value|value.as_ref().ok_or_else(||"bounded batch expert output is absent".into())).collect::<Result<Vec<_>>>()?;
            let output=compute.join(&refs)?;values.insert(plan.steps[index+stage].outputs[0].name.clone(),output);
        }Ok(true)
    }
    fn independent_batch(&self,index:usize,plan:&LayerPlan,values:&mut BTreeMap<String,C::Value>)->Result<Vec<usize>> {
        let first=&plan.steps[index];if first.inputs.len()!=1 {return Ok(Vec::new());}
        let source=&first.inputs[0];let mut selected=Vec::new();
        for (stage,step) in plan.steps.iter().enumerate().skip(index) {
            if step.outputs.iter().any(|output|&output.name==source) {break;}
            if step.kind!=Kind::Project || step.inputs!=first.inputs || step.weights.len()!=1 || step.outputs.len()!=1 {continue;}
            let output=&step.outputs[0].name;
            if plan.steps.iter().any(|candidate|candidate.kind==Kind::Route && candidate.inputs.contains(output))
                || plan.steps.iter().filter(|candidate|candidate.outputs.iter().any(|buffer|&buffer.name==output)).count()!=1 {continue;}
            selected.push(stage);if selected.len()==9 {break;}
        }
        if selected.len()<2 || selected[0]!=index {return Ok(Vec::new());}
        let value=values.get(source).ok_or("bounded independent projection input is absent")?;
        let input=selected.iter().map(|&stage|(plan.steps[stage].weights[0].name.as_str(),value)).collect::<Vec<_>>();
        let Some(output)=self.compute.projection_batch(self.checkpoint,&input,true)? else {return Ok(Vec::new());};
        need(output.len()==selected.len(),"bounded independent projection output count differs")?;
        for (&stage,value) in selected.iter().zip(output) {values.insert(plan.steps[stage].outputs[0].name.clone(),value);}
        Ok(selected)
    }
    fn expert_rows(&self,index:usize,plans:&[LayerPlan],values:&mut [BTreeMap<String,C::Value>])->Result<()> {
        let b=&self.compute;let mut groups:BTreeMap<usize,Vec<(usize,usize)>>=BTreeMap::new();
        let mut outputs:Vec<Vec<Option<C::Value>>>=Vec::new();
        for (row,plan) in plans.iter().enumerate() {
            let step=&plan.steps[index];let ids=b.read(&values[row][&step.inputs[1]])?;let mut seen=Vec::new();
            outputs.push((0..ids.len()).map(|_|None).collect());
            for (slot,id) in ids.into_iter().enumerate() {
                need(id>=0.0 && id.fract()==0.0 && (id as usize)<step.weights.len(),"bounded expert ID is outside recipe")?;
                let id=id as usize;need(!seen.contains(&id),"bounded route repeats an expert")?;seen.push(id);
                groups.entry(id).or_default().push((row,slot));
            }
        }
        for (id,rows) in groups {
            let name=&plans[0].steps[index].weights[id].name;
            need(rows.iter().all(|(row,_)|plans[*row].steps[index].weights[id].name==*name),"prompt expert weight binding differs")?;
            let owned=rows.iter().map(|(row,slot)| {let step=&plans[*row].steps[index];let width=step.weights[id].shape[1] as usize;
                if step.kind==Kind::ExpertDown {b.range(&values[*row][&step.inputs[0]],slot*width,width).map(Some)} else {Ok(None)}
            }).collect::<Result<Vec<_>>>()?;
            let refs=rows.iter().zip(&owned).map(|((row,_),value)|value.as_ref().unwrap_or(&values[*row][&plans[*row].steps[index].inputs[0]])).collect::<Vec<_>>();let projected=self.project_rows(name,&refs,true)?;
            need(projected.len()==rows.len(),"prompt expert output count differs")?;
            for ((row,slot),value) in rows.into_iter().zip(projected) {outputs[row][slot]=Some(value);}
        }
        for (row,output) in outputs.into_iter().enumerate() {
            let refs=output.iter().map(|value|value.as_ref().ok_or_else(||"prompt expert output is absent".into())).collect::<Result<Vec<_>>>()?;
            let value=b.join(&refs)?;values[row].insert(plans[row].steps[index].outputs[0].name.clone(),value);
        }Ok(())
    }

}
#[path="bounded_decoder/rows.rs"] mod rows;
pub use rows::{HeadRows,PROMPT_GROUP,EXPANDED_ROWS,expanded_bytes,group_workspace_bytes};
#[path="bounded_decoder/host.rs"] mod host;
pub use host::Host;
#[cfg(target_os="linux")]
#[path="bounded_decoder/device.rs"] mod device;
#[cfg(target_os="linux")]
pub use device::{Device,Value};
#[cfg(target_os="linux")]
#[path="bounded_decoder/factory.rs"] mod factory;
#[cfg(target_os="linux")]
pub use factory::{Factory,ResidentFactory};
