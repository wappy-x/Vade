//! Bounded causal recipe interpreter. Full service admission stays separate.
#![allow(clippy::chunks_exact_to_as_chunks, clippy::collapsible_if, clippy::needless_borrow, clippy::needless_late_init, reason = "kept as on main while lane-byte-cut edits this file; fix after it merges")]
use crate::{Result,causal_layer,mxfp8_projection::Mxfp8Projection,layer_driver::LayerBoundary};
use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan,LayerSchedule,Topology},state::StatePlan,storage::WeightStorage,description::TokenMixer},safetensors::Dtype,flash::{nvfp4,activation_nvfp4::Activation}};
use spark_gpu_probe::{causal_ops::{Shape,Op},norm_boundary::{NormShape,WeightType}};
use std::{cell::RefCell,collections::BTreeMap,rc::Rc};
#[path="bounded_decoder/parameters.rs"] mod parameters;
pub use parameters::{StaticParameters,ParameterCounts};
use parameters::StaticParameterPlan;
pub const PROMPT_ROWS:usize=4;
fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
pub(crate) fn greedy_logits(logits:&[f32],words:usize)->Result<u32> {
    need(!logits.is_empty() && logits.len()==words && logits.iter().all(|value|value.is_finite()),"bounded logits extent or value differs")?;
    let mut best=(logits[0],0);for (id,&value) in logits.iter().enumerate().skip(1) {
        best=spark_gpu_probe::next_math::better(best,(value,u32::try_from(id).map_err(|_|"bounded logits ID overflow")?));
    }Ok(best.1)
}
/// CHAIN-SHADOW target: the first static projection that reads the normalized output of the
/// mHC boundary at `boundary` (its prepare, collapse and norm steps), before the next boundary.
/// Expert projections never match: their rows depend on routing. Returns the weight name.
pub fn shadow_target(steps:&[engine_model::plan::layer::Step],boundary:usize)->Option<&str> {shadow_targets(steps,boundary).into_iter().next()}
/// Every such static projection of that boundary, in plan order (the order the stage reads them).
pub fn shadow_targets(steps:&[engine_model::plan::layer::Step],boundary:usize)->Vec<&str> {
    let Some(output)=steps.get(boundary+2).and_then(|s|s.outputs.first()).map(|o|&o.name) else {return Vec::new()};
    steps.get(boundary+3..).unwrap_or(&[]).iter().take_while(|s|s.kind!=Kind::MhcPrepare)
        .filter(|s|s.kind==Kind::Project && s.inputs.first()==Some(output) && s.weights.len()==1).map(|s|s.weights[0].name.as_str()).collect()
}
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
#[cfg_attr(not(target_os="linux"),allow(dead_code))]
mod memo_batches;
mod residual_plan;
/// Per-step allocation fields, with the names RESIDENT VERIFY prints: device vectors
/// newly allocated between two cumulative counts (count, bytes). Print only.
pub fn allocation_fields(before:Option<(usize,usize)>,now:Option<(usize,usize)>)->String {
    match (before,now) {
        (Some(b),Some(n))=>format!("new_buffers={} new_bytes={}",n.0.saturating_sub(b.0),n.1.saturating_sub(b.1)),
        _=>"new_buffers=none new_bytes=none".into(),
    }
}
/// Prompt-group allocation fields: the step fields plus `reused_buffers`, the vector
/// pool's spares taken again between two cumulative counts (count, bytes, reused). Print only.
pub fn group_allocation_fields(before:Option<(usize,usize,usize)>,now:Option<(usize,usize,usize)>)->String {
    match (before,now) {
        (Some(b),Some(n))=>format!("{} reused_buffers={}",allocation_fields(Some((b.0,b.1)),Some((n.0,n.1))),n.2.saturating_sub(b.2)),
        _=>"new_buffers=none new_bytes=none reused_buffers=none".into(),
    }
}
pub trait Compute {
    /// One ordered execution owns temporary buffers through state publication.
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {run()}
    fn resident_weights_ready(&self)->bool {false}
    fn fatal_error(&self)->Option<String> {None}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {None}
    /// Diagnostic: device vectors newly allocated so far, as (count, bytes). None off the device.
    fn vector_allocations(&self)->Option<(usize,usize)> {None}
    /// Diagnostic: (new count, new bytes, spares reused) so far. None off the device.
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {None}
    fn serving_trace(&self)->bool {false}
    fn profile_tick_begin(&self,_tick:u64) {}
    fn profile_tick_end(&self) {}
    fn plan_cost(&self,_ns:u128) {}
    fn trace_start(&self) {}
    fn cost_report(&self,_phase:&str) {}
    type Value;type Pass;
    fn projects_complete_rows(&self)->bool {false}
    fn sparse_topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn owned_expansion(&self,_:&str,_:&Self::Value,_:usize,_:crate::sparse_heads::Layout)->Result<Option<Self::Value>> {Ok(None)}
    fn owned_expansion_rows(&self,_:&str,_:&Self::Value,_:usize,_:crate::sparse_heads::Layout)->Result<Option<Self::Value>> {Ok(None)}
    fn gather_attended(&self,value:&Self::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Self::Value> {
        need(layout.local_heads()==layout.total_heads(),"compute has no attended gather")?;
        self.range(value,0,tokens*layout.total_heads()*layout.value_dim())
    }
    fn prepare(&self,_:&CheckpointAdmission)->Result<()> {Ok(())}
    /// More resident matrices of the prepared checkpoint (the appended predictor layer).
    fn prepare_names(&self,_:&CheckpointAdmission,_:&[String])->Result<()> {Ok(())}
    fn projection(&self,_:&CheckpointAdmission,_:&str,_:&Self::Value,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    fn greedy_projection(&self,c:&CheckpointAdmission,name:&str,x:&Self::Value)->Result<u32> {
        let head=Mxfp8Projection::matrix(c,name,WeightType::F32)?;let values=self.mx(&head,x)?;
        greedy_logits(&self.read(&values)?,head.shape().rows())
    }
    fn projection_rows(&self,_:&CheckpointAdmission,_:&str,_:&[&Self::Value],_:bool)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn projection_batch(&self,_:&CheckpointAdmission,_:&[(&str,&Self::Value)],_:bool)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn projection_stage(&self,_:&CheckpointAdmission,_:&[crate::projection_stage::Job<'_,Self::Value>])->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn routed_experts(&self,_:&[engine_model::plan::layer::Step],_:&Self::Value,_:&Self::Value)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// A kept projection of exactly these immutable rows. No device read.
    fn projection_origin(&self,_:&str,_:&Self::Value,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    /// `projection_origin` for rows `slots` of `rows` (each `width` words), in that order. A
    /// device answers all of them in one call and makes no view of a row it does not keep.
    fn projection_origins(&self,name:&str,rows:&Self::Value,width:usize,slots:&[usize],round:bool)->Result<Vec<Option<Self::Value>>> {
        slots.iter().map(|&slot|{let row=self.range(rows,slot.checked_mul(width).ok_or("kept row offset overflow")?,width)?;self.projection_origin(name,&row,round)}).collect()
    }
    /// Keep pure projections (input rows, output) for later tokens. The owner
    /// bounds the kept bytes; all pairs of one call come from one allocation.
    fn remember_projections(&self,_:&str,_:bool,_:&[(&Self::Value,&Self::Value)])->Result<()> {Ok(())}
    fn repeated_projection(&self,c:&CheckpointAdmission,name:&str,x:&Self::Value,round:bool)->Result<Option<Self::Value>> {self.projection(c,name,x,round)}
    fn attention(&self,_heads:u32,_key_dim:u32,_value_dim:u32,_slots:usize,_capacity:usize,_x:[&Self::Value;4])->Result<Option<Self::Value>> {Ok(None)}
    fn indirect_attention(&self,_shape:Shape,_parts:&[&Self::Value],_:&Self::Value)->Result<Option<Self::Value>> {Ok(None)}
    /// True when this compute runs the planar form of a KDA step or convolution (Op::StepPlanar, Op::ConvParts) for this shape: one
    /// launch whose result holds all heads' states then all outputs (or all convolution outputs then all histories), taken as two views.
    fn planar_kda(&self,_shape:Shape)->bool {false}
    fn embedding(&self,_:u32,_:usize)->Result<Option<Self::Value>> {Ok(None)}
    fn upload(&self,v:&[f32])->Result<Self::Value>;
    fn read(&self,v:&Self::Value)->Result<Vec<f32>>;
    fn read_pair(&self,first:&Self::Value,second:&Self::Value)->Result<(Vec<f32>,Vec<f32>)> {Ok((self.read(first)?,self.read(second)?))}
    /// `row` is the whole selection output, known without its bytes (select_admission::known_selection).
    /// A backend may skip reading these values, if it still compares the count word before any result leaves.
    /// No default: a wrapper that drops this call would silently lose the saving.
    fn select_known(&self,ids:&Self::Value,count:&Self::Value,row:&[f32])->Result<()>;
    fn range(&self,v:&Self::Value,start:usize,len:usize)->Result<Self::Value>;
    /// A join whose storage has one size for every call of its kind (`reserve` words, the
    /// most the join can hold), so the storage of an earlier call can serve again.
    fn join_reserved(&self,parts:&[&Self::Value],_reserve:usize)->Result<Self::Value> {self.join(parts)}
    fn join(&self,parts:&[&Self::Value])->Result<Self::Value>;
    /// A scalar latent-cache append may keep bounded room beyond its immutable prefix.
    fn cache_append(&self,old:&Self::Value,new:&Self::Value,_rows:usize,_width:usize,_room:usize)->Result<Self::Value> {self.join(&[old,new])}

    fn zeros(&self,len:usize)->Result<Self::Value>;
    fn operation(&self,s:Shape,x:[Option<&Self::Value>;4],w:&[f32])->Result<Self::Value>;
    /// Gather keeps its logical extent but reuses storage at the session's slot bound.
    fn gather_reserved(&self,s:Shape,x:[Option<&Self::Value>;4],_reserve:usize)->Result<Self::Value> {self.operation(s,x,&[])}
    /// Static weights retain their original upload owner; inputs and shape remain per call.
    fn operation_prepared(&self,s:Shape,x:[Option<&Self::Value>;4],w:&Rc<StaticParameters>)->Result<Self::Value> {self.operation(s,x,w.values())}
    fn norm(&self,name:&str,s:NormShape,x:&Self::Value,w:&[f32])->Result<Self::Value>;
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value)->Result<(Self::Pass,Self::Value)>;
    /// `begin` for one decode row. `next` names the static projections that read this
    /// boundary's output, in plan order; a device may pull their weights toward L2 while the
    /// chain runs (CHAIN-SHADOW). The values and their order are those of `begin`.
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value,_next:&[&str])->Result<(Self::Pass,Self::Value)> {self.begin(key,p,x)}
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
    /// A large group has ended: its spare working memory can go.
    fn rows_release(&self)->Result<()> {Ok(())}
    /// After a prompt group: keep its spares within the pool's normal room (default: release).
    fn rows_keep(&self)->Result<()> {self.rows_release()}
    /// One prompt group's peak device bytes: begin returns a token for end. Zero off the device.
    fn rows_peak_begin(&self)->usize {0}
    fn rows_peak_end(&self,_whole:usize)->usize {0}
    /// Room for this many words, to be written by `rows_fill` before any read.
    fn rows_reserve(&self,words:usize)->Result<Self::Value> {self.zeros(words)}
    /// Write `part` at `offset` in a value only this caller holds.
    fn rows_fill(&self,target:&mut Self::Value,offset:usize,part:&Self::Value,expected:usize)->Result<()> {
        let (mut all,part)=(self.read(target)?,self.read(part)?);need(part.len()==expected,"prompt fill extent differs")?;
        all.get_mut(offset..offset+part.len()).ok_or("prompt fill exceeds its value")?.copy_from_slice(&part);
        *target=self.upload(&all)?;Ok(())
    }
    /// Earlier expanded rows plus these.
    fn rows_append(&self,held:Option<&Self::Value>,_:usize,new:&Self::Value,_:usize)->Result<Self::Value> {
        match held {Some(old)=>self.join(&[old,new]),None=>self.range(new,0,self.read(new)?.len())}
    }
    /// `rows_append` into storage with room for `room` more words. A later call whose held rows are
    /// the longest live part of that storage writes its new rows there and copies nothing.
    fn rows_extend(&self,held:Option<&Self::Value>,rows:usize,new:&Self::Value,width:usize,_room:usize)->Result<Self::Value> {self.rows_append(held,rows,new,width)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&Self::Value,m:usize,w:&[f32])->Result<Self::Value>;
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value,m:usize)->Result<(Self::Pass,Self::Value)>;
    fn rows_finish(&self,p:Self::Pass,x:&Self::Value,m:usize)->Result<Self::Value>;
}
/// `expanded` keeps each prompt layer's expanded latent rows. Only prompt groups
/// use it, and it holds only pure projections of `latent_cache` rows.
pub struct Expanded<V>(pub std::rc::Rc<V>,pub usize,pub crate::sparse_heads::Layout);
impl<V> Clone for Expanded<V> {fn clone(&self)->Self {Self(self.0.clone(),self.1,self.2)}}
pub struct State<V> {pub position:u64,pub layers:Vec<BTreeMap<String,V>>,pub expanded:BTreeMap<usize,Expanded<V>>}
/// Host copies of immutable parameters, shared by all sessions of one server.
#[derive(Clone)]
/// Immutable parameters of one checkpoint. They stay bound to it: a decoder built
/// from them reads that checkpoint's weights and can read no other.
pub struct Prepared<'f> {checkpoint:&'f CheckpointAdmission,parameters:Rc<BTreeMap<String,Vec<f32>>>,static_parameters:Rc<StaticParameterPlan>,blocks:Rc<BTreeMap<(usize,usize),crate::layer_operands::ResidualBlockOperands>>,schedules:Vec<LayerSchedule>,draft:Option<(Rc<proposer::Draft>,LayerSchedule)>}
pub struct Decoder<'f,C:Compute> {checkpoint:&'f CheckpointAdmission,compute:C,capacity:u64,parameters:Rc<BTreeMap<String,Vec<f32>>>,static_parameters:Rc<StaticParameterPlan>,
    blocks:Rc<BTreeMap<(usize,usize),crate::layer_operands::ResidualBlockOperands>>,schedules:RefCell<Vec<LayerSchedule>>,expanded_limit:std::cell::Cell<usize>,forced_union:std::cell::Cell<Option<usize>>,
    draft:Option<Rc<proposer::Draft>>,draft_schedule:RefCell<Option<LayerSchedule>>}
impl<'f,C:Compute> Decoder<'f,C> {
    fn sparse_layout(&self,layer:usize)->Result<crate::sparse_heads::Layout> {
        let mixer=self.checkpoint.description().layers.get(layer).or_else(||self.draft.as_ref().and_then(|draft|draft.description.layers.get(layer))).map(|spec|spec.mixer).ok_or("sparse layer is absent")?;
        let TokenMixer::LatentAttention(attention)=mixer else {return Err("sparse layout requires latent attention".into());};
        crate::sparse_heads::Layout::new(attention.heads,attention.key_dim,attention.value_dim,self.compute.sparse_topology())
    }
    fn expand_owned(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<C::Value> {
        if let Some(value)=self.compute.owned_expansion(name,input,tokens,layout)? {return Ok(value);}
        need(layout.local_heads()==layout.total_heads(),"compute has no owned expansion")?;
        if tokens==1 {self.project(name,input,true)} else {self.project_group(name,input,tokens,true)}
    }
    fn local_query(&self,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<C::Value> {
        let rows=layout.query_rows();let width=layout.total_heads()*layout.key_dim();
        if rows.len()==width {return self.compute.range(input,0,tokens*width);}
        if tokens==1 {return self.compute.range(input,rows.start,rows.len());}
        let parts=(0..tokens).map(|token|self.compute.range(input,token*width+rows.start,rows.len())).collect::<Result<Vec<_>>>()?;
        self.compute.join(&parts.iter().collect::<Vec<_>>())
    }
    fn expand_owned_rows(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<C::Value> {
        if let Some(value)=self.compute.owned_expansion_rows(name,input,tokens,layout)? {return Ok(value);}
        need(layout.local_heads()==layout.total_heads(),"compute has no owned row expansion")?;
        self.project_group(name,input,tokens,true)
    }
    pub fn new(checkpoint:&'f CheckpointAdmission,compute:C,capacity:u64)->Result<Self> {
        let d=checkpoint.description();need(!d.layers.is_empty() && d.hidden>0 && d.residual.streams>0 && d.vocab>0 && d.vocab<=u32::MAX as u64 && !d.tied_embeddings,"unsupported bounded text geometry")?;
        need(capacity>0 && capacity<=checkpoint.description().max_positions,"bounded decoder capacity exceeds the recipe")?;
        let mut parameter_jobs=Vec::new();let mut parameters=BTreeMap::new();let mut blocks=BTreeMap::new();let mut schedules=Vec::new();let topology=Topology {rank:0,parts:1};
        let storage=StatePlan::new(d,topology,capacity,isize::MAX as usize).map_err(|e|e.to_string())?;
        for layer in 0..d.layers.len() {let boundary=LayerBoundary::new(d,checkpoint.headers(),&storage,layer,1,topology,usize::MAX)?;
            residual_plan::check(boundary.program())?;
            schedules.push(LayerSchedule::new(d,layer,topology).map_err(|e|e.to_string())?);
            for (index,step) in boundary.program().steps.iter().enumerate() {
                if StaticParameterPlan::admits(step) {parameter_jobs.push((layer,step.id.clone(),step.weights.iter().map(|w|w.name.clone()).collect()));}
                if step.kind==Kind::MhcPrepare {blocks.insert((layer,index),boundary.prepare_residual_block_checkpoint(index,checkpoint)?);}
                for w in &step.weights {if w.storage==WeightStorage::Plain && !parameters.contains_key(&w.name) {parameters.insert(w.name.clone(),plain(checkpoint,&w.name)?);}}
            }
        }
        let name="model.language_model.norm.weight";parameters.insert(name.into(),plain(checkpoint,name)?);
        let parameters=Rc::new(parameters);let static_parameters=Rc::new(StaticParameterPlan::new(parameters.clone(),parameter_jobs)?);
        compute.prepare(checkpoint)?;Ok(Self {checkpoint,compute,capacity,parameters,static_parameters,blocks:Rc::new(blocks),schedules:RefCell::new(schedules),expanded_limit:std::cell::Cell::new(rows::RESERVED_ROWS),forced_union:std::cell::Cell::new(None),draft:None,draft_schedule:RefCell::new(None)})
    }
    pub fn prepared(&self)->Prepared<'f> {Prepared {checkpoint:self.checkpoint,parameters:self.parameters.clone(),static_parameters:self.static_parameters.clone(),blocks:self.blocks.clone(),schedules:self.schedules.borrow().clone(),draft:self.draft.clone().zip(self.draft_schedule.borrow().clone())}}
    /// A new session without rereading immutable parameters.
    pub fn from_prepared(prepared:&Prepared<'f>,compute:C,capacity:u64)->Result<Self> {
        let checkpoint=prepared.checkpoint;
        need(capacity>0 && capacity<=checkpoint.description().max_positions,"bounded decoder capacity exceeds the recipe")?;
        compute.prepare(checkpoint)?;
        Ok(Self {checkpoint,compute,capacity,parameters:prepared.parameters.clone(),static_parameters:prepared.static_parameters.clone(),blocks:prepared.blocks.clone(),schedules:RefCell::new(prepared.schedules.clone()),expanded_limit:std::cell::Cell::new(rows::RESERVED_ROWS),forced_union:std::cell::Cell::new(None),draft:prepared.draft.as_ref().map(|d|d.0.clone()),draft_schedule:RefCell::new(prepared.draft.as_ref().map(|d|d.1.clone()))})
    }
    pub fn zero_state(&self)->Result<State<C::Value>> {self.zero_state_at(0)}
    /// Zero state of the shape a chat has at this position. Position zero is a new chat; any
    /// other position is for memory tests, because zeros are not what a prompt leaves there.
    pub fn zero_state_at(&self,position:u64)->Result<State<C::Value>> {
        let mut layers=Vec::new();for layer in 0..self.checkpoint.description().layers.len() {
            let p=LayerPlan::new(self.checkpoint.description(),layer,position+1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;let mut inputs=BTreeMap::new();
            for b in p.inputs.iter().filter(|b|b.name!="residual" && b.name!="local_slots") {inputs.insert(b.name.clone(),self.compute.zeros(b.elements().map_err(|e|e.to_string())?)?);}layers.push(inputs);
        }Ok(State {position,layers,expanded:BTreeMap::new()})
    }
    /// No committed state changes until every decoder layer and the head succeed.
    pub fn step(&self,token:u32,state:&State<C::Value>,observe:impl FnMut(usize,&C::Value)->Result<()>)->Result<(Vec<f32>,u32,State<C::Value>)> {
        self.step_hidden(token,state,observe).map(|(logits,best,next,_)|(logits,best,next))
    }
    /// The same step, with the row's last hidden state before and after the final norm.
    /// This is the plain decode step: its mHC boundaries name their next stage (CHAIN-SHADOW).
    pub fn step_hidden(&self,token:u32,state:&State<C::Value>,observe:impl FnMut(usize,&C::Value)->Result<()>)->Result<(Vec<f32>,u32,State<C::Value>,[C::Value;2])> {
        self.step_purpose(token,state,observe,true)
    }
    pub fn step_greedy(&self,token:u32,state:&State<C::Value>,observe:impl FnMut(usize,&C::Value)->Result<()>)->Result<(u32,State<C::Value>)> {
        let (next,hidden)=self.step_body(token,state,observe,true)?;
        let best=self.compute.greedy_projection(self.checkpoint,"lm_head.weight",&hidden[1])?;
        need(u64::from(best)<self.checkpoint.description().vocab,"bounded greedy ID exceeds vocabulary")?;Ok((best,next))
    }
    /// One row. `shadow` is true only for the plain decode step; verification and prompt rows
    /// pass no next stage, so their boundaries launch the grids they had before CHAIN-SHADOW.
    pub(crate) fn step_purpose(&self,token:u32,state:&State<C::Value>,observe:impl FnMut(usize,&C::Value)->Result<()>,shadow:bool)->Result<(Vec<f32>,u32,State<C::Value>,[C::Value;2])> {
        let (next,hidden)=self.step_body(token,state,observe,shadow)?;
        let head=Mxfp8Projection::matrix(self.checkpoint,"lm_head.weight",WeightType::F32)?;let values=self.compute.mx(&head,&hidden[1])?;
        let logits=self.compute.read(&values)?;let best=greedy_logits(&logits,self.checkpoint.description().vocab as usize)?;
        Ok((logits,best,next,hidden))
    }
    fn step_body(&self,token:u32,state:&State<C::Value>,mut observe:impl FnMut(usize,&C::Value)->Result<()>,shadow:bool)->Result<(State<C::Value>,[C::Value;2])> {
        let c=self.checkpoint;let d=c.description();need(state.position<self.capacity && state.layers.len()==d.layers.len(),"bounded state context differs")?;
        need(u64::from(token)<d.vocab,"bounded input token exceeds vocabulary")?;
        let mut residual=self.embedding(token)?;let mut layers=Vec::new();
        for layer in 0..d.layers.len() {let (mut values,next)=self.layer_group(layer,state.position+1,&[&residual],&state.layers[layer],state.expanded.get(&layer),shadow)?;let output=values.remove(0);observe(layer,&output)?;residual=output;layers.push(next);}
        let h=d.hidden as usize;let mean=self.compute.mean(&residual,number(d.residual.streams)?,number(d.hidden)?)?;
        let norm_name="model.language_model.norm.weight";let normalized=self.compute.norm(norm_name,norm_shape(c,norm_name,h)?,&mean,&self.parameters[norm_name])?;
        Ok((State {position:state.position+1,layers,expanded:state.expanded.clone()},[mean,normalized]))
    }
    /// Layer-major prompt rows. State producers run in ascending token order;
    /// each matrix row retains the scalar feature fold and rounding boundaries.
    pub fn prefill(&self,tokens:&[u32],state:&State<C::Value>)->Result<(Vec<Vec<f32>>,State<C::Value>)> {
        need(!tokens.is_empty() && tokens.len()<=PROMPT_ROWS && state.layers.len()==self.checkpoint.description().layers.len(),"bounded prompt group differs")?;
        need(state.position.checked_add(tokens.len() as u64).is_some_and(|n|n<=self.capacity),"bounded prompt exceeds capacity")?;
        let mut residuals=tokens.iter().map(|&token|self.embedding(token)).collect::<Result<Vec<_>>>()?;let mut layers=Vec::new();
        for layer in 0..self.checkpoint.description().layers.len() {
            let (next,state)=self.layer_group(layer,state.position+1,&residuals.iter().collect::<Vec<_>>(),&state.layers[layer],None,false)?;
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
    /// Pack expanded K/V words two BF16 patterns a word. Every producer here passes round=true, so each
    /// word is already BF16-rounded and packing keeps its exact pattern (PackBf16 on a rounded word is the
    /// identity on its high half). Stored rows (memo, held, selected union) are all packed; no other form exists.
    fn pack_expanded(&self,value:&C::Value,words:usize)->Result<C::Value> {
        need(words>0 && words%2==0,"packed expanded rows need an even word count")?;
        let mut s=shape(Op::PackBf16);s.rows=number((words/2) as u64)?;
        self.compute.operation(s,[Some(value),None,None,None],&[])
    }
    /// The unpacked form, for the join-and-UnpackKv fallback only. Exact inverse of `pack_expanded`.
    fn unpack_expanded(&self,value:&C::Value,packed_words:usize)->Result<C::Value> {
        need(packed_words>0,"packed expanded rows are empty")?;
        let mut s=shape(Op::UnpackBf16);s.rows=number(packed_words as u64)?;
        self.compute.operation(s,[Some(value),None,None,None],&[])
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
        let (mut values,state)=self.layer_group(layer,context,&[residual],state,None,false)?;
        Ok((values.remove(0),state))
    }
    fn layer_group(&self,layer:usize,first_context:u64,residuals:&[&C::Value],state:&BTreeMap<String,C::Value>,expanded:Option<&Expanded<C::Value>>,shadow:bool)->Result<(Vec<C::Value>,BTreeMap<String,C::Value>)> {
        self.layer_steps(layer,first_context,residuals,state,expanded,false,shadow)
    }
    /// `state_only`: the steps run until the layer's next state is complete (the caches a later
    /// position reads). What follows them, the attention itself and the experts, is left out,
    /// and no output is returned. For a row whose output nobody reads.
    /// `shadow`: the plain decode step (one row); see `step_purpose`.
    fn layer_steps(&self,layer:usize,first_context:u64,residuals:&[&C::Value],state:&BTreeMap<String,C::Value>,expanded:Option<&Expanded<C::Value>>,state_only:bool,shadow:bool)->Result<(Vec<C::Value>,BTreeMap<String,C::Value>)> {
        if let Some(owner)=expanded {need(owner.2==self.sparse_layout(layer)?,"expanded owner layout differs")?;}
        need(!residuals.is_empty() && residuals.len()<=PROMPT_ROWS,"bounded prompt row count differs")?;
        let count=residuals.len();let c=self.checkpoint;let d=c.description();let b=&self.compute;
        need(first_context>0 && first_context.checked_add(count as u64-1).is_some_and(|n|n<=self.capacity && n<=u32::MAX as u64),"bounded layer context exceeds capacity")?;
        let planning=std::time::Instant::now();
        // The appended predictor layer follows the decoder layers. It has one residual stream:
        // its prepare is the norm alone and its combine is a BF16 add (see proposer.rs).
        let drafting=layer==d.layers.len();let mixer=self.mixer(layer)?;let streams=if drafting {1} else {d.residual.streams as usize};
        let mut main=self.schedules.borrow_mut();let mut extra=self.draft_schedule.borrow_mut();
        let schedule=if drafting {extra.as_mut().ok_or("predictor layer is not enabled")?} else {main.get_mut(layer).ok_or("bounded layer is outside recipe")?};
        let owned_plans:Vec<LayerPlan>;
        let plans:Vec<&LayerPlan>=if count==1 {vec![schedule.at(first_context).map_err(|e|e.to_string())?]} else {
            owned_plans=(0..count).map(|row|schedule.at(first_context+row as u64).cloned().map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
            owned_plans.iter().collect()
        };
        b.plan_cost(planning.elapsed().as_nanos());
        let names=match mixer {
            TokenMixer::Kda(_)=>vec![("kda.history","next_history"),("kda.state","next_state")],
            // The complete pools ride with the state. The state plan already holds their plane.
            TokenMixer::LatentAttention(_)=>vec![("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates"),("pool_keys","pool_keys")],
        };
        let mut all_values=(0..count).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        for (values,residual) in all_values.iter_mut().zip(residuals) {values.insert("residual".to_owned(),b.range(residual,0,d.hidden as usize*streams)?);}
        let mut all_passes=(0..count).map(|_|BTreeMap::new()).collect::<Vec<_>>();let mut skip=0;let mut projected=std::collections::BTreeSet::new();
        for index in 0..plans[0].steps.len() {
            if state_only && names.iter().all(|(_,output)|all_values[count-1].contains_key(*output)) {break;}
            if skip>0 {skip-=1;continue;}
            if projected.remove(&index) {continue;}
            let first=&plans[0].steps[index];
            need(plans.iter().all(|p|p.steps.len()==plans[0].steps.len() && p.steps[index].kind==first.kind && p.steps[index].id==first.id),"prompt row programs differ")?;
            if count==1 && matches!(first.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown) {
                let done=self.projection_stage(index,&plans[0],&mut all_values[0])?;
                if !done.is_empty() {projected.extend(done);continue;}
            }
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
                if drafting && step.kind==Kind::MhcPrepare {let target=&plan.steps[index+2];let name=&target.weights[0].name;
                    let value=b.norm(name,norm_shape(c,name,d.hidden as usize)?,&values[&step.inputs[0]],self.parameter(name)?)?;values.insert(target.outputs[0].name.clone(),value);skip=2;continue;
                }
                if drafting && step.kind==Kind::Combine {let value=self.bf16_sum(&values[&step.inputs[0]],&values[&step.inputs[2]])?;values.insert(step.outputs[0].name.clone(),value);continue;}
                if step.kind==Kind::MhcPrepare {
                    let key=(layer,index);let p=self.blocks.get(&key).ok_or("resident residual parameters are absent")?;
                    let next=if shadow && count==1 {shadow_targets(&plan.steps,index)} else {Vec::new()};
                    let (pass,norm)=b.begin_next(key,p,&values[&step.inputs[0]],&next)?;
                    passes.insert(step.id.split('.').next().unwrap(),pass);values.insert(plan.steps[index+2].outputs[0].name.clone(),norm);skip=2;continue;
                }
                if step.kind==Kind::Combine {let side=step.id.split('.').next().unwrap();let pass=passes.remove(side).ok_or("bounded residual pass is absent")?;
                    let value=b.finish(pass,&values[&step.inputs[2]])?;values.insert(step.outputs[0].name.clone(),value);continue;}
            if count==1 && step.kind==Kind::Scores {
                if let (Some(softmax),Some(value))=(plan.steps.get(index+1),plan.steps.get(index+2)) {
                    let private=plan.returns.iter().all(|name|name!=&step.outputs[0].name && name!=&softmax.outputs[0].name) && plan.steps[index+3..].iter().all(|later|later.inputs.iter().all(|name|name!=&step.outputs[0].name && name!=&softmax.outputs[0].name));
                    if softmax.kind==Kind::Softmax && value.kind==Kind::Values && private && softmax.inputs[0]==step.outputs[0].name && value.inputs[0]==softmax.outputs[0].name && softmax.inputs[1]==step.inputs[2] && value.inputs[2]==step.inputs[2] {
                        if let TokenMixer::LatentAttention(a)=mixer {
                            let slots=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}).min(context) as usize;
                            let capacity=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}).min(self.capacity) as usize;
                            let layout=self.sparse_layout(layer)?;let query=self.local_query(&values[&step.inputs[0]],1,layout)?;
                            let args=[&query,&values[&step.inputs[1]],&values[&value.inputs[1]],&values[&step.inputs[2]]];
                            if let Some(output)=b.attention(number(layout.local_heads() as u64)?,number(a.key_dim)?,number(a.value_dim)?,slots,capacity,args)? {
                                values.insert(value.outputs[0].name.clone(),b.gather_attended(&output,1,layout)?);skip=2;continue;
                            }
                        }
                    }
                }
            }
            let input=step.inputs.iter().filter(|n|n.as_str()!="local_slots").map(|n|values.get(n).ok_or_else(||format!("missing bounded input {n}"))).collect::<Result<Vec<_>>>()?;
            let output:Vec<C::Value>=match step.kind {
                Kind::Norm=>{let name=&step.weights[0].name;let width=step.outputs[0].elements().map_err(|e|e.to_string())?;
                    vec![b.norm(name,norm_shape(c,name,width)?,input[0],self.parameter(name)?)?]},
                Kind::CacheAppend=>{let TokenMixer::LatentAttention(a)=mixer else {return Err("cache append requires latent attention".into());};
                    let width=a.kv_rank as usize;
                    let room=((self.capacity-context).min(HELD_ROOM as u64) as usize).checked_mul(width).ok_or("latent cache room overflow")?;
                    vec![if count==1 {b.cache_append(input[0],input[1],context as usize-1,width,room)?} else {b.join(&[input[0],input[1]])?}]},
                Kind::ExpandKv=>{
                    let TokenMixer::LatentAttention(a)=mixer else {return Err("KV expansion requires latent recipe".into());};
                    let (selected,selected_ids)=if expanded.is_some() {
                        let (ids,count)=b.read_pair(values.get("selected").ok_or("bounded selection is absent")?,input[1])?;(count[0],Some(ids))
                    } else {(b.read(input[1])?[0],None)};
                    need(selected>=0.0,"selection score is not a number")?;
                    let count=selected as usize;let slots=step.outputs[0].shape[0].min(context) as usize;
                    let layout=self.sparse_layout(layer)?;
                    need(count<=slots,"bounded KV count exceeds slots")?;let (h,kd,vd,lc)=(layout.local_heads(),a.key_dim as usize,a.value_dim as usize,a.kv_rank as usize);
                    if slots==0 {vec![b.zeros(0)?,b.zeros(0)?]} else {
                    let mut projected=Vec::new();
                    // Prompt groups already hold pure expansions of their rows. A run of
                    // consecutive selected rows needs no latent lookup. The copy kernel's grid
                    // is parts times the longest part, so runs stay at one pool of rows.
                    let row=h*(kd+vd);need(kd%2==0 && vd%2==0,"packed expanded rows need even key and value dims")?;let packed_row=row/2;let ids=selected_ids.unwrap_or_default();
                    // Rows that are neither held nor kept are expanded together: one
                    // projection for all of them, as a prompt group does, and no read of
                    // a latent row. The expansion of a row depends only on its bits.
                    // One pass lists the runs of held rows and the rows the memo is asked for. The memo
                    // answers those in one call. The rows it lacks are expanded next, and the views of
                    // the held runs are made last: the device expands while the host lists them. The
                    // launches, their order and every operand are those of a slot-by-slot walk.
                    let name=&step.weights[0].name;let mut missing=Vec::new();let (mut runs,mut asked)=(Vec::new(),Vec::new());
                    let mut slot=0;while slot<count {
                        if let Some(Expanded(_,held,_))=expanded {let id=ids[slot] as usize;if id<*held {
                            let mut end=slot+1;while end<count && end-slot<a.indexer.pool as usize && ids[end] as usize==id+(end-slot) && (ids[end] as usize)<*held {end+=1;}
                            runs.push((projected.len(),id,end-slot));projected.push(None);slot=end;continue;}}
                        asked.push((projected.len(),slot));projected.push(None);slot+=1;}
                    let found=b.projection_origins(name,input[0],lc,&asked.iter().map(|&(_,slot)|slot).collect::<Vec<_>>(),true)?;
                    need(found.len()==asked.len(),"kept expansion count differs")?;
                    for (&(at,slot),value) in asked.iter().zip(found) {
                        match value {Some(value)=>projected[at]=Some(value),None=>missing.push((at,b.range(input[0],slot*lc,lc)?))}
                    }
                    for group in missing.chunks(PROMPT_GROUP) {
                        // One new row keeps the single-row path and its device exchange.
                        // A group is filled to one of a few sizes with copies of its last
                        // row, so its working buffers have sizes the allocator saw before.
                        // A row's expansion does not depend on the other rows of its group.
                        let (rows,filled)=if group.len()==1 {(self.expand_owned(name,&group[0].1,1,layout)?,1usize)} else {
                            let filled=if group.len()<=64 {group.len().next_power_of_two()} else {group.len().div_ceil(64)*64};
                            let mut latents=group.iter().map(|(_,latent)|latent).collect::<Vec<_>>();
                            let last=group.last().map(|(_,latent)|latent).ok_or("bounded expansion group is empty")?;latents.resize(filled,last);
                            (self.expand_owned(name,&b.join(&latents)?,filled,layout)?,filled)};
                        // Pack at the producer: the rounded projection is scratch from here; every stored row is packed.
                        let rows=self.pack_expanded(&rows,filled*row)?;
                        for (index,(at,_)) in group.iter().enumerate() {projected[*at]=Some(b.range(&rows,index*packed_row,packed_row)?);}
                        let kept=group.iter().map(|(at,latent)|projected[*at].as_ref().map(|value|(latent,value)).ok_or("bounded expansion is absent")).collect::<std::result::Result<Vec<_>,_>>()?;
                        b.remember_projections(name,true,&kept)?;
                    }
                    if let Some(Expanded(rows,_,_))=expanded {for &(at,id,len) in &runs {projected[at]=Some(b.range(rows,id*packed_row,len*packed_row)?);}}
                    let projected=projected.into_iter().map(|value|value.ok_or_else(||"bounded expansion is absent".to_string())).collect::<Result<Vec<_>>>()?;
                    if residuals.len()==1 {
                        if let Some((query,output))=indirect::attention_tail(plan,index) {
                            let mut s=shape(Op::ScoresIndirect);s.heads=number(h as u64)?;s.dim=number(kd as u64)?;s.cols=number(vd as u64)?;s.rows=number(slots as u64)?;s.taps=number(count as u64)?;
                            let query=self.local_query(values.get(query).ok_or("indirect query is absent")?,1,layout)?;
                            if let Some(result)=b.indirect_attention(s,&projected.iter().collect::<Vec<_>>(),&query)? {
                                values.insert(output.to_owned(),b.gather_attended(&result,1,layout)?);skip=3;continue;
                            }
                        }
                    }
                    let joined=b.join_reserved(&projected.iter().collect::<Vec<_>>(),slots*packed_row)?;
                    // The fallback keeps its unpacked planes: widen the packed rows back, bit for bit.
                    let plain=if count==0 {b.zeros(0)?} else {self.unpack_expanded(&joined,count*packed_row)?};
                    let mut s=shape(Op::UnpackKv);s.rows=number(slots as u64)?;s.heads=number(h as u64)?;s.dim=number(kd as u64)?;
                    s.cols=number(vd as u64)?;s.taps=number(count as u64)?;
                    let unpacked=b.operation(s,[Some(&plain),None,None,None],&[])?;let keys=slots*h*kd;
                    vec![b.range(&unpacked,0,keys)?,b.range(&unpacked,keys,slots*h*vd)?]}
                },
                kind=>{
                    let mut s=shape(match kind {Kind::Conv=>Op::Conv,Kind::KdaPrepare=>Op::Prepare,Kind::KdaStep=>Op::Step,Kind::GatedNorm=>Op::GatedNorm,
                        Kind::SwiGlu=>Op::SwiGlu,Kind::KeyNorm=>Op::KeyNorm,Kind::Pool=>Op::Pool,Kind::ScorePools=>Op::PoolScores,Kind::Select=>Op::Select,
                        Kind::Gather=>Op::Gather,Kind::Scores=>Op::Scores,Kind::Softmax=>Op::Softmax,Kind::Values=>Op::Values,Kind::Route=>Op::Route,Kind::ExpertSum=>Op::ExpertFold,
                        _=>return Err(format!("{}: bounded operation is unsupported",step.id))});
                    s.epsilon=d.epsilon as f32;
                    match mixer {
                        TokenMixer::Kda(k)=>{s.heads=number(k.heads)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;if kind==Kind::KdaPrepare {s.bound=k.gate_lower_bound as f32;}},
                        TokenMixer::LatentAttention(a)=>{
                            s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;
                            match kind {
                                Kind::KeyNorm=>s.epsilon=plan.settings.index_key_epsilon,
                                Kind::Pool=>s.rows=number(context)?,
                                Kind::ScorePools=>s.rows=number(context/a.indexer.pool)?,
                                Kind::Select=>{s.rows=number(context)?;s.taps=number(a.indexer.token_budget)?;s.dim=number(a.indexer.pool)?;s.cols=u32::from(a.indexer.select_tail);},
                                Kind::Gather=>{s.rows=number(context)?;s.dim=number(a.kv_rank)?;s.cols=number(step.outputs[0].shape[0].min(context))?;},
                                Kind::Scores|Kind::Softmax|Kind::Values=>{s.heads=number(self.sparse_layout(layer)?.local_heads() as u64)?;s.dim=number(if kind==Kind::Values {a.value_dim} else {a.key_dim})?;
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
                    let prepared=self.static_parameters.get(layer,&self.parameters,step);
                    let mut weight=Vec::new();if prepared.is_none() {for w in &step.weights {weight.extend_from_slice(self.parameter(&w.name)?);}}
                    let weight=prepared.map_or(weight.as_slice(),|p|p.values());
                    let operation=|s,x|match prepared {Some(p)=>b.operation_prepared(s,x,p),None=>b.operation(s,x,weight)};
                    // A complete pool never changes: it folds only its own rows. The state
                    // keeps the complete pools of the earlier rows, so one token appends its
                    // key and gate and folds at most the one pool that it completes.
                    if let Some(prior)=state.get("pool_keys").filter(|_|kind==Kind::Pool && count==1 && context>1) {
                        let (dim,pool,rows)=(s.dim as usize,s.taps as usize,context as usize);
                        // One key row and one gate row a token: the funded append writes them past the
                        // held prefix (same words, same order as the join) and copies only at growth.
                        let room=((self.capacity-context).min(HELD_ROOM as u64) as usize).checked_mul(dim).ok_or("index cache room overflow")?;
                        let keys=b.cache_append(input[0],input[2],rows-1,dim,room)?;let gates=b.cache_append(input[1],input[3],rows-1,dim,room)?;
                        let earlier=b.range(prior,0,((rows-1)/pool)*dim)?;
                        let pools=if rows%pool!=0 {earlier} else {
                            let mut last=s;last.rows=number(pool as u64)?;
                            let old_keys=b.range(input[0],(rows-pool)*dim,(pool-1)*dim)?;let old_gates=b.range(input[1],(rows-pool)*dim,(pool-1)*dim)?;
                            let folded=operation(last,[Some(&old_keys),Some(&old_gates),Some(input[2]),Some(input[3])])?;
                            b.join(&[&earlier,&b.range(&folded,2*pool*dim,dim)?])?
                        };
                        need(step.outputs.len()==3,"bounded pool output count differs")?;
                        for (spec,v) in step.outputs.iter().zip([keys,gates,pools]) {values.insert(spec.name.clone(),v);}
                        continue;
                    }
                    // Planar forms: one launch, two views of its result, no packing copy before the Conv and no joins after either op.
                    let planar=match kind {Kind::Conv|Kind::KdaStep=>{let mut p=s;p.op=if kind==Kind::Conv {Op::ConvParts} else {Op::StepPlanar};if b.planar_kda(p) {Some(p)} else {None}},_=>None};
                    let packed;if kind==Kind::Conv && planar.is_none() {packed=Some(b.join(&input[..3])?);} else {packed=None;}
                    // Select retains its complete diagnostic row. Gather reads only the
                    // causal prefix; padded slots never enter a live reduction.
                    let selected=if kind==Kind::Gather {Some(b.range(input[1],0,s.cols as usize)?)} else {None};
                    let args=if let Some(v)=selected.as_ref() {[Some(input[0]),Some(v),Some(input[2]),None]}
                        else if let Some(v)=packed.as_ref() {[Some(v),Some(input[3]),None,None]} else {std::array::from_fn(|i|input.get(i).copied())};
                    if let Some(p) = planar {
                        let result=operation(p,args)?;
                        let (hd,width)=(p.heads as usize*p.dim as usize,p.heads as usize*p.dim as usize);
                        let pair=if kind==Kind::Conv {
                            let history=3*(p.taps as usize-1)*width;
                            vec![b.range(&result,0,3*width)?,b.range(&result,3*width,history)?]
                        } else {
                            let state=hd*p.dim as usize;
                            vec![b.range(&result,state,hd)?,b.range(&result,0,state)?]
                        };
                        need(pair.len()==step.outputs.len(),"bounded output count differs")?;
                        for (spec,v) in step.outputs.iter().zip(pair) {values.insert(spec.name.clone(),v);}
                        continue;
                    }
                    // The same total order and output layout without the repeated scans.
                    let mut known_row=None;
                    let result=if kind==Kind::Select {
                        let groups=context/s.dim as u64;let mut row=spark_gpu_probe::prompt_ops::Shape::new(spark_gpu_probe::prompt_ops::Op::SelectRows,1).ok_or("selection row count")?;
                        row.dim=s.dim;row.taps=s.taps;row.rows=number(groups)?;row.cols=s.cols;row.first=number(context)?;
                        let TokenMixer::LatentAttention(a)=mixer else {return Err("selection requires a latent recipe".into());};
                        row.slots=number(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0})?;
                        // The count read below refuses a NaN score, with no extra wait here.
                        known_row=spark_gpu_probe::select_admission::known_selection(row);
                        b.prompt(row,[args[0],None,None,None,None],&[])?
                    } else if kind==Kind::Gather {
                        let reserve=(step.outputs[0].shape[0].min(self.capacity) as usize).checked_mul(s.dim as usize).ok_or("gather capacity overflow")?;
                        b.gather_reserved(s,args,reserve)?
                    } else if kind==Kind::Scores {
                        let query=self.local_query(input[0],1,self.sparse_layout(layer)?)?;operation(s,[Some(&query),args[1],args[2],args[3]])?
                    } else {operation(s,args)?};
                    let result=if kind==Kind::Values {b.gather_attended(&result,1,self.sparse_layout(layer)?)?} else {result};
                    let (_,_,units,words)=s.lengths().ok_or("invalid bounded operation shape")?;
                    match kind {
                        Kind::Conv|Kind::KdaStep=>{
                            let first=if kind==Kind::Conv {s.heads as usize*s.dim as usize} else {s.dim as usize*s.dim as usize};
                            let mut a=Vec::new();let mut c=Vec::new();for i in 0..units {a.push(b.range(&result,i*words,first)?);c.push(b.range(&result,i*words+first,words-first)?);}
                            let a=b.join(&a.iter().collect::<Vec<_>>())?;let c=b.join(&c.iter().collect::<Vec<_>>())?;
                            if kind==Kind::KdaStep {vec![c,a]} else {vec![a,c]}
                        },
                        Kind::Pool=>{let len=context as usize*s.dim as usize;vec![b.range(&result,0,len)?,b.range(&result,len,len)?,b.range(&result,2*len,(context as usize/s.taps as usize)*s.dim as usize)?]},
                        Kind::Select=>{let slots=words-1;let ids=b.range(&result,0,slots)?;let count=b.range(&result,slots,1)?;
                            if let Some(row)=known_row.as_ref() {b.select_known(&ids,&count,row)?;}vec![ids,count]},
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
        let mut output=Vec::new();if !state_only {for values in &mut all_values {output.push(values.remove("output").ok_or("bounded layer has no output")?);}}
        let last=all_values.last_mut().ok_or("empty prompt layer")?;let mut next=BTreeMap::new();
        for (input,output) in names {next.insert(input.into(),last.remove(output).ok_or("bounded layer has no next state")?);}Ok((output,next))
    }
    fn import_row_state(&self,row:usize,index:usize,plans:&[&LayerPlan],names:&[(&str,&str)],state:&BTreeMap<String,C::Value>,values:&mut [BTreeMap<String,C::Value>])->Result<()> {
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
        let compute=&self.compute;let first=&plan.steps[index];let stages=&plan.steps[index..index+span];
        if stages.iter().all(|step|step.inputs==first.inputs && step.outputs.len()==1 && step.kind==first.kind) {
            if let Some(outputs)=compute.routed_experts(stages,&values[&first.inputs[1]],&values[&first.inputs[0]])? {
                need(outputs.len()==span,"device route output count differs")?;
                for (step,output) in stages.iter().zip(outputs) {values.insert(step.outputs[0].name.clone(),output);}return Ok(true);
            }
        }
        spark_gpu_probe::device_route::fallback();let ids=compute.read(&values[&first.inputs[1]])?;
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
            if values.contains_key(output) {continue;}
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
    fn expert_rows(&self,index:usize,plans:&[&LayerPlan],values:&mut [BTreeMap<String,C::Value>])->Result<()> {
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
pub use rows::{full_memo_ballast,HeadRows,PROMPT_GROUP,PROMPT_GROUP_MAX,PROMPT_SHARED,RESERVED_ROWS,HELD_ROOM,POOL_ROOM,releases_spares_first,PROJECTION_MEMO_BYTES,expanded_bytes,group_workspace_bytes,group_workspace_bytes_for,OpenRows};
mod indirect;
#[path="bounded_decoder/projection_stage.rs"] mod projection_stage;
#[cfg(any(target_os="linux",test))] #[path="bounded_decoder/held.rs"] mod held;
#[path="bounded_decoder/proposer.rs"] mod proposer;
pub use proposer::{PROPOSER_HIDDEN,PredictorState};
#[path="bounded_decoder/verify.rs"] mod verify;
pub use verify::{Verify,CARRIED_ROWS,VERIFY_ROWS,TRAIL_ROWS,PROMPT_TRAIL,learned_pass_ms};
#[path="bounded_decoder/host.rs"] mod host;
pub use host::Host;
#[cfg(target_os="linux")]
#[path="bounded_decoder/device.rs"] mod device;
#[cfg(all(target_os="linux",feature="kernel-bench"))]
#[path="bounded_decoder/projection_probe.rs"] mod projection_probe;
#[cfg(target_os="linux")]
pub use device::{Device,Value,SLAB_ROWS};
#[cfg(target_os="linux")]
#[path="bounded_decoder/factory.rs"] mod factory;
#[cfg(target_os="linux")]
pub use factory::{Factory,ResidentFactory};
