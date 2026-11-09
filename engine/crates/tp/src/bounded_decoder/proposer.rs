//! The checkpoint's appended predictor layer as a proposer of next tokens. Nothing trusts its
//! output: a verify pass keeps a proposal only where plain decoding chooses the same token.
//! The reviewed source does not define this layer (model/src/plan/mtp-source-gap.md). The
//! forward pass here is the public module of this family: fuse the normed embedding of the
//! chosen token with the normed hidden state, one decoder layer with a plain residual, the
//! shared head norm, the vocabulary head. A wrong detail lowers the kept rate and nothing else.
use super::*;
use engine_model::plan::TextDescription;
/// Which of the two hidden states a proposal starts from: 0 before the final norm, 1 after it.
/// Chosen by the measured kept rate (runner flag --propose). Only the kept rate depends on it.
pub const PROPOSER_HIDDEN:usize=1;
/// The predictor layer's own cache of the positions it has run, for one chat.
pub struct PredictorState<V> {rows:u64,layers:BTreeMap<String,V>}
pub struct Draft {pub(super) description:TextDescription,pub(super) parameters:BTreeMap<String,Vec<f32>>}
impl<'f,C:Compute> Decoder<'f,C> {
    /// Load the predictor layer beside the decoder. Without this call nothing of it is read.
    pub fn enable_predictor(&mut self)->Result<()> {
        if self.draft.is_some() {return Ok(());}
        let c=self.checkpoint;let description=c.description().with_draft_layer().map_err(|e|e.to_string())?.ok_or("checkpoint declares no predictor layer")?;
        let layer=c.description().layers.len();let topology=Topology {rank:0,parts:1};
        let plan=LayerPlan::new(&description,layer,1,topology).map_err(|e|e.to_string())?;
        let mut parameters=BTreeMap::new();let mut names=Vec::new();
        for step in plan.steps.iter().filter(|step|step.kind!=Kind::MhcPrepare) {
            let matrix=matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown|Kind::ExpandKv);
            for w in &step.weights {
                if matrix {names.push(w.name.clone());}
                else if w.storage==WeightStorage::Plain && !parameters.contains_key(&w.name) {parameters.insert(w.name.clone(),plain(c,&w.name)?);}
            }
        }
        let prefix=format!("model.language_model.layers.{layer}");
        for norm in ["enorm","hnorm","shared_head.norm"] {let name=format!("{prefix}.{norm}.weight");parameters.insert(name.clone(),plain(c,&name)?);}
        names.push(format!("{prefix}.eh_proj.weight"));
        self.compute.prepare_names(c,&names)?;
        *self.draft_schedule.borrow_mut()=Some(LayerSchedule::new(&description,layer,topology).map_err(|e|e.to_string())?);
        self.draft=Some(Rc::new(Draft {description,parameters}));Ok(())
    }
    pub fn predictor_enabled(&self)->bool {self.draft.is_some()}
    pub(super) fn mixer(&self,layer:usize)->Result<TokenMixer> {
        let d=self.checkpoint.description();
        if let Some(block)=d.layers.get(layer) {return Ok(block.mixer);}
        self.draft.as_ref().and_then(|draft|draft.description.layers.get(layer)).map(|block|block.mixer).ok_or_else(||"bounded layer is outside recipe".into())
    }
    pub(super) fn parameter(&self,name:&str)->Result<&Vec<f32>> {
        self.parameters.get(name).or_else(||self.draft.as_ref().and_then(|draft|draft.parameters.get(name))).ok_or_else(||format!("missing bounded parameter {name}"))
    }
    /// The residual add of a BF16 model: each sum is rounded to BF16.
    pub(super) fn bf16_sum(&self,x:&C::Value,y:&C::Value)->Result<C::Value> {
        let (x,y)=(self.compute.read(x)?,self.compute.read(y)?);need(x.len()==y.len(),"residual sum extent differs")?;
        let sum=x.iter().zip(&y).map(|(a,b)|engine_model::plan::layer::cast::round_bf16(a+b).map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
        self.compute.upload(&sum)
    }
    /// `depth` proposals for the tokens after `token`. `hidden` is the target's last hidden
    /// state of the row before `token`. The layer sees no earlier position (context 1).
    pub fn propose(&self,hidden:&C::Value,token:u32,depth:usize)->Result<Vec<u32>> {self.propose_after(hidden,token,depth,None).map(|(tokens,_)|tokens)}
    /// The same, with the layer's own cache of earlier positions. `seen` is that cache and its
    /// row count; the second result is the cache after this position (the first proposal's
    /// step), for the caller to pass on once `token` is a committed token.
    pub fn propose_after(&self,hidden:&C::Value,token:u32,depth:usize,seen:Option<&PredictorState<C::Value>>)->Result<(Vec<u32>,PredictorState<C::Value>)> {
        self.propose_scored(hidden,token,depth,seen).map(|(tokens,_,after)|(tokens,after))
    }
    /// The same, with two scores of each proposal: its logit minus the second largest logit,
    /// and its probability under the predictor's own distribution.
    pub fn propose_scored(&self,hidden:&C::Value,token:u32,depth:usize,seen:Option<&PredictorState<C::Value>>)->Result<(Vec<u32>,Vec<[f32;2]>,PredictorState<C::Value>)> {
        self.propose_chain(hidden,token,depth,seen,true)
    }
    /// The same, with the form of the chain named. `normalized` (the served form): each deeper step
    /// takes the layer's output after the head norm as its hidden state, as the first step takes
    /// the target's hidden state after the final norm. Counted on four prompts (run s16): both
    /// proposals right 284 -> 295 of 436. The other form passes the output itself, for a count.
    pub fn propose_chain(&self,hidden:&C::Value,token:u32,depth:usize,seen:Option<&PredictorState<C::Value>>,normalized:bool)->Result<(Vec<u32>,Vec<[f32;2]>,PredictorState<C::Value>)> {
        let draft=self.draft.as_ref().ok_or("predictor layer is not enabled")?;let c=self.checkpoint;let d=c.description();let b=&self.compute;
        let h=d.hidden as usize;let layer=d.layers.len();let prefix=format!("model.language_model.layers.{layer}");
        need(depth<=3 && u64::from(token)<d.vocab,"proposal depth or token differs")?;
        let norm=|part:&str,x:&C::Value|->Result<C::Value> {let name=format!("{prefix}.{part}.weight");
            b.norm(&name,norm_shape(c,&name,h)?,x,draft.parameters.get(&name).ok_or("predictor norm is absent")?)};
        let plan=LayerPlan::new(&draft.description,layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
        let mut empty=BTreeMap::new();
        for input in plan.inputs.iter().filter(|input|input.name!="residual" && input.name!="local_slots") {empty.insert(input.name.clone(),b.zeros(input.elements().map_err(|e|e.to_string())?)?);}
        let first=seen.map_or(0,|s|s.rows);need(first+depth.max(1) as u64<=self.capacity,"predictor cache exceeds capacity")?;
        // `after` is the cache with this position; `chain` runs on from it for deeper proposals only.
        let mut after:Option<PredictorState<C::Value>>=None;let mut chain:Option<BTreeMap<String,C::Value>>=None;
        let head=Mxfp8Projection::matrix(c,"lm_head.weight",WeightType::F32)?;
        let mut hidden=b.range(hidden,0,h)?;let mut token=token;let mut proposals=Vec::new();let mut scores=Vec::new();
        // Depth 0 runs the layer once for its cache and makes no proposal.
        for step in 0..depth.max(1) {
            let embedded=b.range(&self.embedding(token)?,0,h)?;
            let fused=b.join(&[&norm("enorm",&embedded)?,&norm("hnorm",&hidden)?])?;
            let x=self.project(&format!("{prefix}.eh_proj.weight"),&fused,true)?;
            let earlier=match (step,&after,&chain) {(0,_,_)=>seen.map_or(&empty,|seen|&seen.layers),(1,Some(after),_)=>&after.layers,(_,_,Some(chain))=>chain,_=>return Err("proposal chain lost its cache".into())};
            // Depth 0 only brings the layer's cache up to this row: its output is not read, so the steps end with the state.
            let (mut outputs,next)=self.layer_steps(layer,first+step as u64+1,&[&x],earlier,None,depth==0,false)?;
            if step==0 {after=Some(PredictorState {rows:first+1,layers:next});} else {chain=Some(next);}
            if depth==0 {break;}
            let output=outputs.remove(0);
            let headed=norm("shared_head.norm",&output)?;let logits=b.read(&b.mx(&head,&headed)?)?;
            need(logits.len()==d.vocab as usize && logits.iter().all(|v|v.is_finite()),"proposal logits extent or value differs")?;
            let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
            let top=logits[best];let (mut second,mut mass)=(f32::NEG_INFINITY,0f64);
            for (i,&v) in logits.iter().enumerate() {if i!=best && v>second {second=v;}mass+=f64::from(v-top).exp();}
            scores.push([top-second,(1.0/mass) as f32]);
            token=best as u32;proposals.push(token);hidden=if normalized {headed} else {output};
        }Ok((proposals,scores,after.ok_or("proposal made no step")?))
    }
}
