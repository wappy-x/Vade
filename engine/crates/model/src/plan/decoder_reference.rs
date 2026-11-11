//! Causal scalar FP32 decoder reference. No checkpoint casts or device admission.
//! Callers supply decoded logical weights. This cannot serve a checkpoint.
use super::{TextDescription, layer::{Buffer, Element, LayerPlan, Topology,
    reference::{self, Data, Tensor, Tensors, WeightSource}}, description::TokenMixer};
use crate::{Result, invalid};
use std::sync::Arc;

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const NORM: &str = "model.language_model.norm.weight";
const HEAD: &str = "lm_head.weight";
fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn tensor(weights: &dyn WeightSource, name: &str, shape: &[u64]) -> Result<Tensor> {
    let t=weights.load(name)?;
    need(t.shape==shape,"decoder reference weight shape differs")?;
    let count=Buffer::new(name,shape).elements()?;
    let values=t.floats()?;
    need(values.len()==count && values.iter().all(|x| x.is_finite()),"invalid decoder reference weight values")?;
    Ok(t)
}
fn zeros(shape: &[u64]) -> Result<Tensor> {
    let count=Buffer::new("state",shape).elements()?;
    let mut values=Vec::new();
    values.try_reserve_exact(count).map_err(|_| invalid("cannot reserve decoder reference state"))?;
    values.resize(count,0.0); Tensor::f32(shape,values)
}

/// Full-vocabulary logical results from one causal token. All values are FP32.
/// `next_token` uses the lowest token ID for ties. Nonfinite logits fail.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub context: u64, pub input_token: u32, pub residual: Tensor,
    pub hidden: Tensor, pub normalized: Tensor, pub logits: Tensor, pub next_token: u32,
}
/// A private copy of the committed causal boundary. It retains its owner ID.
#[derive(Clone)]
pub struct Saved { owner: Arc<()>, context: u64, state: Vec<Tensors> }
impl Saved { pub fn context(&self) -> u64 { self.context } }

/// One host reference lane. Model data stays borrowed and immutable.
/// The scalar layer reference fixes arithmetic independently of device topology.
pub struct Decoder<'a> {
    description: &'a TextDescription, weights: &'a dyn WeightSource, capacity: u64,
    context: u64, state: Vec<Tensors>, owner: Arc<()>,
}
impl<'a> Decoder<'a> {
    /// Check all logical weight shapes before the first append. Values load
    /// on demand. The source must retain its checkpoint identity across calls.
    /// Storage recipes do not imply that decoded values match a loaded runtime.
    pub fn new(description: &'a TextDescription, weights: &'a dyn WeightSource, capacity: u64) -> Result<Self> {
        weights.profile().require_scalar().map_err(|e| invalid(e.to_string()))?;
        let d=description;
        need(capacity>0 && capacity<=d.max_positions && capacity<=u32::MAX as u64
            && !d.layers.is_empty(),"invalid decoder reference capacity or empty stack")?;
        need(d.vocab>0 && d.vocab<=u32::MAX as u64,"invalid decoder reference vocabulary")?;
        for (name,shape) in [(EMBEDDING,vec![d.vocab,d.hidden]),(NORM,vec![d.hidden]),
            (if d.tied_embeddings {EMBEDDING} else {HEAD},vec![d.vocab,d.hidden])] {
            need(weights.shape(name)?==shape,"decoder reference weight shape differs")?;
        }
        let mut state=Vec::new();
        for layer in 0..d.layers.len() {
            let p=LayerPlan::new(d,layer,1,Topology {parts:1,rank:0})?;
            LayerPlan::new(d,layer,capacity,Topology {parts:1,rank:0})?;
            for w in p.steps.iter().flat_map(|s| &s.weights) { need(weights.shape(&w.name)?==w.shape,"decoder reference weight shape differs")?; }
            let mut inputs=Tensors::new();
            for b in &p.inputs {
                if b.name=="residual" { continue; }
                let value=match b.element {
                    Element::F32=>zeros(&b.shape)?,
                    Element::U32=>Tensor {shape:b.shape.clone(),data:Data::U32((0..b.elements()?)
                        .map(|j| u32::try_from(j).map_err(|_| invalid("reference slot exceeds token ABI")))
                        .collect::<Result<Vec<_>>>()?)},
                };
                inputs.insert(b.name.clone(),value);
            }
            state.push(inputs);
        }
        Ok(Self {description,weights,capacity,context:0,state,owner:Arc::new(())})
    }
    pub fn context(&self) -> u64 { self.context }
    pub fn capacity(&self) -> u64 { self.capacity }
    pub fn save(&self) -> Saved { Saved {owner:self.owner.clone(),context:self.context,state:self.state.clone()} }
    /// A snapshot from another lane fails even when its geometry is equal.
    pub fn restore(&mut self, saved: &Saved) -> Result<()> {
        need(Arc::ptr_eq(&self.owner,&saved.owner),"saved decoder reference belongs to another lane")?;
        self.state=saved.state.clone(); self.context=saved.context; Ok(())
    }
    pub fn step(&mut self, token: u32) -> Result<Output> {
        Ok(self.append(&[token])?.pop().unwrap())
    }
    /// Process tokens in causal order. A failed chunk leaves the lane unchanged.
    /// Each output includes full logits. Chunk boundaries do not alter the math.
    pub fn append(&mut self, tokens: &[u32]) -> Result<Vec<Output>> {
        self.append_observed(tokens,|_,_,_| Ok(()))
    }
    /// Visit each complete layer result before state commit. Observer errors
    /// roll back the lane. External observer writes cannot be rolled back here.
    pub fn append_observed(&mut self, tokens: &[u32], mut observe: impl FnMut(u64,usize,&reference::Forward)->Result<()>) -> Result<Vec<Output>> {
        self.weights.profile().require_scalar().map_err(|e| invalid(e.to_string()))?;
        let count=u64::try_from(tokens.len()).map_err(|_| invalid("reference token count overflow"))?;
        need(count<=self.capacity-self.context,"decoder reference append exceeds capacity")?;
        need(tokens.iter().all(|&id| u64::from(id)<self.description.vocab),"decoder reference token is outside vocabulary")?;
        if tokens.is_empty() { return Ok(Vec::new()); }
        let mut state=self.state.clone(); let mut output=Vec::new();
        for (index,&token) in tokens.iter().enumerate() {
            output.push(self.compute(token,self.context+index as u64+1,&mut state,&mut observe)?);
        }
        // Every layer, text edge, and selection has succeeded before commit.
        self.state=state; self.context+=count; Ok(output)
    }
    fn compute(&self, token: u32, context: u64, state: &mut [Tensors],
        observe: &mut impl FnMut(u64,usize,&reference::Forward)->Result<()>) -> Result<Output> {
        let d=self.description; let h=usize::try_from(d.hidden).map_err(|_| invalid("reference hidden extent"))?;
        let streams=usize::try_from(d.residual.streams).map_err(|_| invalid("reference stream extent"))?;
        let embedding=self.weights.row(EMBEDDING,token)?;
        need(embedding.shape==[d.hidden],"decoder reference embedding shape differs")?;
        let row=embedding.floats()?;
        need(row.len()==h && row.iter().all(|v| v.is_finite()),"invalid decoder reference embedding values")?;
        let mut values=Vec::new();
        for _ in 0..streams { values.extend_from_slice(row); }
        let mut residual=Tensor::f32(&[d.residual.streams,d.hidden],values)?;
        for (layer,inputs) in state.iter_mut().enumerate() {
            inputs.insert("residual".into(),residual);
            let result=reference::forward_from(d,layer,context,self.weights,inputs)?;
            let plan=LayerPlan::new(d,layer,context,Topology {parts:1,rank:0})?;
            result.check_plan(&plan)?;
            observe(context,layer,&result)?;
            residual=result.tensors["output"].clone();
            // Retain only causal recurrence inputs in this logical FP32 lane.
            // The pool reference recomputes completed pools from retained rows.
            let pairs: &[(&str,&str)]=match d.layers[layer].mixer {
                TokenMixer::Kda(_)=>&[("kda.state","next_state"),("kda.history","next_history")],
                TokenMixer::LatentAttention(_)=>&[("latent_cache","next_latent_cache"),
                    ("index_keys","next_index_keys"),("index_gates","next_index_gates")],
            };
            inputs.remove("residual");
            for &(input,output) in pairs { inputs.insert(input.into(),result.tensors[output].clone()); }
        }
        let raw=residual.floats()?; let mut mean=vec![0.0;h];
        for row in raw.chunks_exact(h) { for (out,&value) in mean.iter_mut().zip(row) { *out+=value; } }
        for value in &mut mean { *value/=streams as f32; }
        let hidden=Tensor::f32(&[d.hidden],mean)?;
        let norm=tensor(self.weights,NORM,&[d.hidden])?;
        let normalized=reference::rms(hidden.floats()?,norm.floats()?,d.epsilon as f32)?;
        let normalized=Tensor::f32(&[d.hidden],normalized)?;
        let head=if d.tied_embeddings {EMBEDDING} else {HEAD};
        let head=tensor(self.weights,head,&[d.vocab,d.hidden])?;
        let logits=reference::project(&normalized,&head)?;
        let values=logits.floats()?; let mut next=0;
        for id in 1..values.len() { if values[id]>values[next] { next=id; } }
        Ok(Output {context,input_token:token,residual,hidden,normalized,logits,next_token:next as u32})
    }
}
