//! A logical decoder program. Kernel targets name source revisions, not device admission.
//! Shapes describe one token. Repeat the same program in causal token order.
use super::{description::*, storage::WeightStorage, TextDescription};
use crate::{Result, invalid, shapes::{Ffn, ExpertStorage, MAX_SPARKS, MAX_HIDDEN, MAX_HEADS, MAX_HEAD_DIM, MAX_WIDTH, MAX_EXPERTS}};
use std::collections::BTreeMap;
pub mod reference;
pub mod cast;

pub const KDA: &str = "77312564120d1730494290744ac633304c2cd3f6";
pub const MLA: &str = "47b34ff3e828538cf4462c28114f5460b1b80115";
pub const INDEXER: &str = "4e4d2a385624d5aed1351f8dd8ac5126ce443e80";
pub const ROUTER: &str = "db57dfbd03cfb729999d5df8aad1d052ccfd3017";
pub const MHC: &str = "1d2180607ab8d9b953fbc0bcacf40468b46ee7f1";
pub const SWIGLU: &str = "b91ec29faadc84801c9de9949660a2dc9ecbac62";
pub const FP8: &str = "97eaa35fdd7484cfa4c715c263e32e659d2924f5";
pub const BASE: &str = "71dbfff0aff6c12e6fe663b477bd7fe30516a3e6";
/// The saved index key LayerNorm definition supplies its own operator epsilon.
pub const INDEX_KEY_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element { F32, U32 }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer { pub name: String, pub shape: Vec<u64>, pub element: Element }
impl Buffer {
    pub fn new(name: &str, shape: &[u64]) -> Self { Self { name: name.into(), shape: shape.into(), element: Element::F32 } }
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n,&v| n.checked_mul(usize::try_from(v).map_err(|_| invalid("layer dimension exceeds usize"))?)
            .filter(|&n| n <= isize::MAX as usize / 4).ok_or_else(|| invalid("layer buffer extent overflow")))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Weight { pub name: String, pub shape: Vec<u64>, pub storage: WeightStorage, pub split: Split }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind { MhcPrepare, Collapse, Norm, Project, Conv, KdaPrepare, KdaStep, GatedNorm,
    CacheAppend, KeyNorm, Pool, ScorePools, Select, Gather, ExpandKv, Scores, Softmax, Values,
    Route, ExpertProject, SwiGlu, ExpertDown, ExpertSum, Combine }
impl Kind {
    // Counts are (inputs, weights, outputs). Expert counts come from the layer.
    fn arity(self, ffn: Ffn) -> Result<(usize,usize,usize)> {
        Ok(match self {
            Self::MhcPrepare => (1,3,1), Self::Collapse => (2,0,1),
            Self::Norm | Self::Project => (1,1,1), Self::Conv => (4,3,2),
            Self::KdaPrepare => (3,2,1), Self::KdaStep => (2,0,2), Self::GatedNorm => (2,1,1),
            Self::CacheAppend => (2,0,1), Self::KeyNorm => (1,2,1), Self::Pool => (4,1,3),
            Self::ScorePools => (3,0,1), Self::Select => (1,0,2), Self::Gather => (3,0,1),
            Self::ExpandKv => (2,1,2), Self::Scores | Self::Values => (4,0,1), Self::Softmax => (2,0,1),
            Self::Route => (1,1,2), Self::SwiGlu => (2,0,1), Self::Combine => (3,0,1),
            Self::ExpertProject | Self::ExpertDown | Self::ExpertSum => {
                let Ffn::Experts(e)=ffn else { return Err(invalid("expert operation requires an expert layer")); };
                if self==Self::ExpertSum { (if e.shared_width>0 {4} else {3},0,1) }
                else { (2,usize::try_from(e.count).map_err(|_| invalid("expert count exceeds usize"))?,1) }
            },
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target { pub revision: &'static str, pub entries: Vec<&'static str>, pub gap: Option<&'static str> }
fn target(revision: &'static str, entries: &[&'static str]) -> Target { Target { revision, entries: entries.into(), gap: None } }
fn gap(reason: &'static str) -> Target { Target { revision: BASE, entries: Vec::new(), gap: Some(reason) } }

/// A tensor split uses complete units on one axis. A unit can be a whole head.
/// Gather copies rank intervals in global order. It never adds partial sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split { Replicated, Axis { axis: usize, unit: u64 } }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology { pub parts: u32, pub rank: u32 }
impl Topology {
    pub fn validate(self) -> Result<()> {
        need((1..=MAX_SPARKS).contains(&self.parts) && (0..self.parts).contains(&self.rank), "invalid layer topology")
    }
    pub fn interval(self, shape: &[u64], split: Split) -> Result<Option<(usize,u64,u64)>> {
        self.validate()?;
        let Split::Axis { axis, unit } = split else { return Ok(None); };
        let n = *shape.get(axis).ok_or_else(|| invalid("split axis is outside shape"))?;
        let stride = mul(unit,self.parts as u64)?;
        need(unit > 0 && n % stride == 0, "split must own complete equal tensor units")?;
        let width = n / self.parts as u64;
        Ok(Some((axis,self.rank as u64 * width,width)))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub id: String, pub kind: Kind, pub inputs: Vec<String>, pub outputs: Vec<Buffer>, pub weights: Vec<Weight>,
    pub target: Target,
    /// Per-output exceptions for operations with different tensor layouts.
    pub output_splits: Vec<Split>,
    /// Logical scratch whose lifetime is this step. Device packing adds its own layout.
    pub scratch: Vec<Buffer>,
    /// Gather these outputs before the next step. Other outputs keep their split.
    pub gather: Vec<String>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub hidden: u64, pub decoder: DecoderDescription, pub residual: SinkhornDescription,
    pub routing: RoutingDescription, pub epsilon: f64, pub swiglu_limit: f64,
    /// Operator epsilon from the saved key LayerNorm definition, not the RMS setting.
    pub index_key_epsilon: f32,
}
impl Settings {
    pub fn from_description(d: &TextDescription,layer: usize) -> Result<Self> {
        Ok(Self { hidden:d.hidden,decoder:*d.layers.get(layer).ok_or_else(|| invalid("layer index is outside decoder"))?,
            residual:d.residual,routing:d.routing,epsilon:d.epsilon,swiglu_limit:d.swiglu_limit,index_key_epsilon:INDEX_KEY_EPSILON })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct LayerPlan {
    pub layer: usize, pub context: u64, pub topology: Topology,
    pub settings: Settings,
    pub inputs: Vec<Buffer>, pub input_splits: BTreeMap<String,Split>, pub steps: Vec<Step>, pub returns: Vec<String>,
}
fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn mul(a: u64,b: u64) -> Result<u64> { a.checked_mul(b).ok_or_else(|| invalid("layer dimension product overflow")) }
fn add(a: u64,b: u64) -> Result<u64> { a.checked_add(b).ok_or_else(|| invalid("layer dimension sum overflow")) }
fn rows(unit: u64) -> Split { Split::Axis { axis: 0, unit } }

// Shared input validation does not construct an operation schedule.
fn validate_data(d: &TextDescription,layer: usize,context: u64) -> Result<DecoderDescription> {
    let block=*d.layers.get(layer).ok_or_else(|| invalid("layer index is outside decoder"))?;
    need(d.rope.is_none(),"logical layer program requires nonrotary text operations")?;
    need(context > 0 && context <= d.max_positions && context <= u32::MAX as u64,"invalid layer context")?;
    need(d.hidden > 0 && d.hidden <= MAX_HIDDEN && d.residual.streams > 0 && d.residual.streams <= MAX_HEADS
        && d.residual.iterations > 0 && d.residual.iterations <= MAX_WIDTH,"invalid layer dimensions")?;
    for value in [d.epsilon,d.residual.epsilon,d.swiglu_limit] {
        need(value.is_finite() && (value as f32).is_finite() && value as f32 > 0.0,"invalid layer scalar")?;
    }
    match block.mixer {
        TokenMixer::Kda(k) => {
            need(k.heads > 0 && k.heads <= MAX_HEADS && k.head_dim > 0 && k.head_dim <= MAX_HEAD_DIM && k.conv > 0 && k.conv <= MAX_HEADS
                && k.gate_lower_bound.is_finite() && (k.gate_lower_bound as f32).is_finite() && (k.gate_lower_bound as f32) < 0.0,"invalid KDA description")?;
        },
        TokenMixer::LatentAttention(a) => {
            let i=a.indexer;
            need([a.heads,a.query_rank,a.kv_rank,a.key_dim,a.value_dim,i.heads,i.dim,i.pool,i.token_budget].iter().all(|&v| v > 0)
                && a.heads <= MAX_HEADS && [a.query_rank,a.kv_rank].iter().all(|v| (1..=MAX_WIDTH).contains(v)) && a.key_dim <= MAX_HEAD_DIM
                && a.value_dim <= MAX_HEAD_DIM && i.heads <= MAX_HEADS && i.dim <= MAX_HEAD_DIM && i.pool <= u32::MAX as u64
                && i.token_budget <= u32::MAX as u64 && a.rotary_dim == 0 && i.learned_pool && i.token_budget % i.pool == 0,"invalid NoPE latent description")?;
            need(add(i.token_budget,if i.select_tail {i.pool-1} else {0})? <= u32::MAX as u64,"selected slot count exceeds device ABI")?;
        },
    }
    match block.ffn {
        Ffn::Dense {width} => need(width > 0 && width <= MAX_WIDTH,"invalid feed-forward width")?,
        Ffn::Experts(e) => {
            need(e.storage==ExpertStorage::PerExpert && e.count > 0 && e.count <= MAX_EXPERTS && e.selected > 0 && e.selected <= e.count
                && e.width > 0 && e.width <= MAX_WIDTH && e.shared_width <= MAX_WIDTH && d.routing.groups > 0
                && e.count % d.routing.groups == 0 && e.count / d.routing.groups >= 2 && d.routing.selected_groups > 0
                && d.routing.selected_groups <= d.routing.groups && e.selected <= d.routing.selected_groups * (e.count / d.routing.groups)
                && d.routing.kind==crate::host_ops::router::RouterKind::CorrectedSigmoid && !d.routing.shared_gate && d.routing.scale.is_finite()
                && (d.routing.scale as f32).is_finite() && d.routing.scale as f32 > 0.0,"invalid corrected routing description")?;
        },
    }
    Ok(block)
}

impl LayerPlan {
    /// Build without reading payloads or allocating model-sized activation buffers.
    /// `context` includes the current token. Cache input planes contain earlier tokens.
    pub fn new(d: &TextDescription, layer: usize, context: u64, topology: Topology) -> Result<Self> {
        topology.validate()?;
        let block=validate_data(d,layer,context)?;
        let mut b = Builder { d, prefix: format!("model.language_model.layers.{layer}"),
            plan: Self { layer,context,topology,settings:Settings::from_description(d,layer)?,inputs: vec![Buffer::new("residual",&[d.residual.streams,d.hidden])],input_splits:BTreeMap::new(),steps: Vec::new(),returns:vec!["output".into()] } };
        b.hc("attn","residual")?;
        b.norm("input_norm","attn.collapsed","input_layernorm",d.hidden);
        match block.mixer { TokenMixer::Kda(k) => b.kda(k)?, TokenMixer::LatentAttention(a) => b.latent(a)? }
        b.combine("attn","residual","attention","after_attention");
        b.hc("ffn","after_attention")?;
        b.norm("post_norm","ffn.collapsed","post_attention_layernorm",d.hidden);
        match block.ffn {
            Ffn::Dense { width } => b.ffn("dense","mlp","post_norm",width,"ffn_output")?,
            Ffn::Experts(e) => {
                b.project("router_logits","post_norm","mlp.gate",e.count,Split::Replicated,false)?;
                let bias = b.weight("mlp.gate.e_score_correction_bias",&[e.count],false);
                let mut ids = Buffer::new("expert_ids",&[e.selected]); ids.element = Element::U32;
                b.op("route",Kind::Route,&["router_logits"],vec![ids,Buffer::new("expert_weights",&[e.selected])],vec![bias],
                    target(ROUTER,&["corrected_sigmoid_route"]),Split::Replicated,&[]);
                b.experts(e.count,e.selected,e.width)?;
                if e.shared_width > 0 { b.ffn("shared","mlp.shared_experts","post_norm",e.shared_width,"shared_output")?; }
                let inputs = if e.shared_width > 0 { vec!["expert_ids","expert_weights","expert_down","shared_output"] }
                    else { vec!["expert_ids","expert_weights","expert_down"] };
                let t=if d.hidden%128==0 && e.shared_width>0 { target(ROUTER,&["ungated_moe_finish"]) }
                    else { gap("ungated expert finish needs aligned hidden rows and a shared row") };
                b.op("ffn_output",Kind::ExpertSum,&inputs,vec![Buffer::new("ffn_output",&[d.hidden])],vec![],t,Split::Replicated,&[]);
            },
        }
        b.combine("ffn","after_attention","ffn_output","output");
        b.plan.validate()?; Ok(b.plan)
    }
    /// Check operation arity, data dependencies, and extents.
    pub fn validate(&self) -> Result<()> {
        self.topology.validate()?;
        let mut buffers = BTreeMap::new();
        let mut placements=BTreeMap::new();
        for buffer in &self.inputs { buffer.elements()?; self.topology.interval(&buffer.shape,self.input_splits.get(&buffer.name).copied().unwrap_or(Split::Replicated))?;
            placements.insert(buffer.name.clone(),self.input_splits.get(&buffer.name).copied().unwrap_or(Split::Replicated));
            need(buffers.insert(buffer.name.clone(),buffer.clone()).is_none(),"duplicate layer input")?; }
        for name in self.input_splits.keys() { need(buffers.contains_key(name),"split input is absent")?; }
        let mut ids = std::collections::BTreeSet::new();
        for step in &self.steps {
            need((step.inputs.len(),step.weights.len(),step.outputs.len())==step.kind.arity(self.settings.decoder.ffn)?,
                "layer operation arity differs")?;
            need(ids.insert(&step.id),"duplicate layer step")?;
            for name in &step.inputs { need(buffers.contains_key(name),&format!("layer input {name} is not ready"))?; }
            if matches!(step.kind,Kind::Project|Kind::ExpertProject|Kind::ExpertDown|Kind::ExpandKv|Kind::MhcPrepare|Kind::Collapse
                |Kind::Combine|Kind::Norm|Kind::Route|Kind::ExpertSum|Kind::KeyNorm|Kind::Pool|Kind::ScorePools|Kind::Select|Kind::Gather) {
                for name in &step.inputs { need(placements[name]==Split::Replicated,&format!("{} requires a gathered input {name}",step.id))?; }
            }
            for w in &step.weights { Buffer::new(&w.name,&w.shape).elements()?; self.topology.interval(&w.shape,w.split)?;
                need(matches!(w.split,Split::Replicated|Split::Axis {axis:0,..}),"weight split must retain complete input features")?; }
            for b in &step.scratch { b.elements()?; }
            need(step.output_splits.len()==step.outputs.len(),"output split count differs")?;
            for (out,&split) in step.outputs.iter().zip(&step.output_splits) {
                out.elements()?; self.topology.interval(&out.shape,split)?;
                placements.insert(out.name.clone(),if step.gather.contains(&out.name) {Split::Replicated} else {split});
                need(buffers.insert(out.name.clone(),out.clone()).is_none(),"duplicate layer output")?;
            }
            for name in &step.gather { need(step.outputs.iter().any(|b| &b.name == name),"gather output is absent")?; }
        }
        for name in &self.returns { need(buffers.contains_key(name),"returned layer buffer is absent")?; }
        need(self.returns.iter().any(|n| n=="output"),"layer output is absent")
    }
    pub fn missing(&self) -> Vec<(&str,&str)> {
        self.steps.iter().filter_map(|s| s.target.gap.map(|g| (s.id.as_str(),g))).collect()
    }
    /// Placement visible to later steps, after each declared gather completes.
    pub fn placements(&self) -> BTreeMap<String,Split> {
        let mut out=self.inputs.iter().map(|b| (b.name.clone(),self.input_splits.get(&b.name).copied().unwrap_or(Split::Replicated))).collect::<BTreeMap<_,_>>();
        for step in &self.steps { for (b,&split) in step.outputs.iter().zip(&step.output_splits) {
            out.insert(b.name.clone(),if step.gather.contains(&b.name) { Split::Replicated } else { split });
        } } out
    }
    /// Birth and inclusive last use for inputs and outputs. Scratch is not included.
    /// External inputs have no birth step. Returns use steps.len() as their last use.
    pub fn lifetimes(&self) -> BTreeMap<String,(Option<usize>,usize)> {
        let mut out: BTreeMap<_,_> = self.inputs.iter().map(|b| (b.name.clone(),(None,0))).collect();
        for (i,s) in self.steps.iter().enumerate() {
            for n in &s.inputs { if let Some(v) = out.get_mut(n) { v.1=i; } }
            for b in &s.outputs { out.insert(b.name.clone(),(Some(i),i)); }
        }
        for n in &self.returns { if let Some(v)=out.get_mut(n) { v.1=self.steps.len(); } }
        out
    }
}

struct Builder<'a> { d: &'a TextDescription, prefix: String, plan: LayerPlan }
impl Builder<'_> {
    fn weight(&self, name: &str, shape: &[u64], projection: bool) -> Weight {
        let module = format!("{}.{}",self.prefix,name);
        Weight { name: if projection { format!("{module}.weight") } else { module.clone() }, shape: shape.into(),
            storage: if projection { self.d.storage.storage(&module) } else { WeightStorage::Plain },split:Split::Replicated }
    }
    fn op(&mut self, id: &str, kind: Kind, inputs: &[&str], outputs: Vec<Buffer>, weights: Vec<Weight>, target: Target, split: Split, gather: &[&str]) {
        let output_splits=vec![split;outputs.len()];
        self.plan.steps.push(Step { id: id.into(),kind,inputs: inputs.iter().map(|s| (*s).into()).collect(),outputs,weights,target,output_splits,scratch:Vec::new(),
            gather: gather.iter().map(|s| (*s).into()).collect() });
    }
    fn project(&mut self, id: &str, input: &str, module: &str, width: u64, split: Split, gather: bool) -> Result<()> {
        let shape = self.plan.inputs.iter().chain(self.plan.steps.iter().flat_map(|s| &s.outputs))
            .find(|b| b.name == input).ok_or_else(|| invalid("projection input is absent"))?.shape.clone();
        let cols = shape.iter().try_fold(1,|a,&b| mul(a,b))?;
        need(width > 0 && cols > 0,"empty projection dimension")?;
        let mut w = self.weight(module,&[width,cols],true); w.split=split;
        let target = weight_target(&w,self.plan.topology)?;
        let gather_outputs=[id];
        self.op(id,Kind::Project,&[input],vec![Buffer::new(id,&[width])],vec![w],target,split,if gather { &gather_outputs } else { &[] });
        Ok(())
    }
    fn norm(&mut self, id: &str, input: &str, module: &str, width: u64) {
        let w = self.weight(&format!("{module}.weight"),&[width],false);
        self.op(id,Kind::Norm,&[input],vec![Buffer::new(id,&[width])],vec![w],target(BASE,&["flash_norm"]),Split::Replicated,&[]);
    }
    fn hc(&mut self, side: &str, input: &str) -> Result<()> {
        let (n,h)=(self.d.residual.streams,self.d.hidden); let c=mul(n,add(n,2)?)?;
        let weights=vec![self.weight(&format!("hc_{side}_fn"),&[c,mul(n,h)?],false),
            self.weight(&format!("hc_{side}_base"),&[c],false),self.weight(&format!("hc_{side}_scale"),&[3],false)];
        let coeff=format!("{side}.coefficients");
        self.op(&format!("{side}.prepare"),Kind::MhcPrepare,&[input],vec![Buffer::new(&coeff,&[c])],weights,
            target(MHC,&["mhc_prepare"]),Split::Replicated,&[]);
        self.plan.steps.last_mut().unwrap().scratch.push(Buffer::new(&format!("{side}.normalized"),&[mul(n,h)?]));
        self.op(&format!("{side}.collapse"),Kind::Collapse,&[input,&coeff],vec![Buffer::new(&format!("{side}.collapsed"),&[h])],vec![],
            target(MHC,&["mhc_collapse"]),Split::Replicated,&[]); Ok(())
    }
    fn combine(&mut self, side: &str, input: &str, block: &str, output: &str) {
        self.op(&format!("{side}.combine"),Kind::Combine,&[input,&format!("{side}.coefficients"),block],
            vec![Buffer::new(output,&[self.d.residual.streams,self.d.hidden])],vec![],target(MHC,&["mhc_combine"]),Split::Replicated,&[]);
    }
    fn ffn(&mut self, id: &str, module: &str, input: &str, width: u64, output: &str) -> Result<()> {
        let (gate,up,mid)=(format!("{id}.gate"),format!("{id}.up"),format!("{id}.middle"));
        self.project(&gate,input,&format!("{module}.gate_proj"),width,rows(1),false)?;
        self.project(&up,input,&format!("{module}.up_proj"),width,rows(1),false)?;
        self.op(&mid,Kind::SwiGlu,&[&gate,&up],vec![Buffer::new(&mid,&[width])],vec![],target(SWIGLU,&["layer_swiglu"]),rows(1),&[&mid]);
        self.project(output,&mid,&format!("{module}.down_proj"),self.d.hidden,rows(1),true)
    }
    fn experts(&mut self, count: u64, selected: u64, width: u64) -> Result<()> {
        for name in ["gate","up"] {
            let weights=(0..count).map(|i| { let mut w=self.weight(&format!("mlp.experts.{i}.{name}_proj"),&[width,self.d.hidden],true); w.split=rows(1); w }).collect::<Vec<_>>();
            let t=expert_target(&weights,self.plan.topology)?;
            let id=format!("expert_{name}");
            self.op(&id,Kind::ExpertProject,&["post_norm","expert_ids"],vec![Buffer::new(&id,&[selected,width])],weights,t,
                Split::Axis { axis: 1,unit: 1 },&[]);
        }
        self.op("expert_middle",Kind::SwiGlu,&["expert_gate","expert_up"],vec![Buffer::new("expert_middle",&[selected,width])],vec![],
            target(SWIGLU,&["layer_swiglu"]),Split::Axis { axis: 1,unit: 1 },&["expert_middle"]);
        let weights=(0..count).map(|i| { let mut w=self.weight(&format!("mlp.experts.{i}.down_proj"),&[self.d.hidden,width],true); w.split=rows(1); w }).collect::<Vec<_>>();
        let t=expert_target(&weights,self.plan.topology)?;
        self.op("expert_down",Kind::ExpertDown,&["expert_middle","expert_ids"],vec![Buffer::new("expert_down",&[selected,self.d.hidden])],weights,t,
            Split::Axis { axis: 1,unit: 1 },&["expert_down"]); Ok(())
    }
    fn kda(&mut self, k: KdaDescription) -> Result<()> {
        let w=mul(k.heads,k.head_dim)?;
        self.plan.topology.interval(&[k.heads],rows(1))?;
        self.plan.inputs.push(Buffer::new("kda.state",&[k.heads,k.head_dim,k.head_dim]));
        self.plan.inputs.push(Buffer::new("kda.history",&[3,k.conv-1,w]));
        self.plan.input_splits.insert("kda.state".into(),rows(1));
        self.plan.input_splits.insert("kda.history".into(),Split::Axis {axis:2,unit:k.head_dim});
        self.plan.returns.extend(["next_state".into(),"next_history".into()]);
        for name in ["q","k","v"] { self.project(name,"input_norm",&format!("self_attn.{name}_proj"),w,rows(k.head_dim),false)?; }
        let weights=["q","k","v"].iter().map(|n| { let mut weight=self.weight(&format!("self_attn.{n}_conv1d.weight"),&[w,k.conv],false); weight.split=rows(k.head_dim); weight }).collect();
        let t=if k.conv == 4 { target(KDA,&["flash_gdn_conv","flash_gdn_history"]) } else { gap("causal convolution tap count has no kernel binding") };
        // Packed [Q,K,V] is a logical concatenation; physical repacking uses copy descriptors.
        self.op("conv",Kind::Conv,&["q","k","v","kda.history"],vec![Buffer::new("convolved",&[3,w]),Buffer::new("next_history",&[3,k.conv-1,w])],weights,t,Split::Replicated,&[]);
        self.plan.steps.last_mut().unwrap().output_splits=vec![Split::Axis { axis: 1,unit: k.head_dim },Split::Axis { axis: 2,unit: k.head_dim }];
        self.project("forget_down","input_norm","self_attn.f_a_proj",k.head_dim,Split::Replicated,false)?;
        self.project("forget","forget_down","self_attn.f_b_proj",w,rows(k.head_dim),false)?;
        self.project("beta_logits","input_norm","self_attn.b_proj",k.heads,rows(1),false)?;
        let mut alog=self.weight("self_attn.A_log",&[k.heads],false); alog.split=rows(1);
        let mut dt=self.weight("self_attn.dt_bias",&[w],false); dt.split=rows(k.head_dim);
        let weights=vec![alog,dt];
        let t=if k.head_dim == 128 { target(KDA,&["flash_kda_prepare"]) } else { gap("KDA prepare requires a supported head dimension") };
        self.op("kda.prepare",Kind::KdaPrepare,&["convolved","forget","beta_logits"],
            vec![Buffer::new("prepared",&[k.heads,4*k.head_dim+1])],weights,t,rows(1),&[]);
        let t=if k.head_dim == 128 { target(KDA,&["flash_kda_step"]) } else { gap("KDA recurrence requires a supported head dimension") };
        self.op("kda.step",Kind::KdaStep,&["prepared","kda.state"],vec![Buffer::new("recurrent",&[k.heads,k.head_dim]),
            Buffer::new("next_state",&[k.heads,k.head_dim,k.head_dim])],vec![],t,rows(1),&[]);
        self.project("output_gate_down","input_norm","self_attn.g_a_proj",k.head_dim,Split::Replicated,false)?;
        self.project("output_gate","output_gate_down","self_attn.g_b_proj",w,rows(k.head_dim),false)?;
        let weight=self.weight("self_attn.o_norm.weight",&[k.head_dim],false);
        let t=if k.head_dim == 128 { target(KDA,&["flash_gdn_gate"]) } else { gap("gated KDA norm requires a supported head dimension") };
        self.op("gated_norm",Kind::GatedNorm,&["recurrent","output_gate"],vec![Buffer::new("gated",&[w])],vec![weight],t,rows(k.head_dim),&["gated"]);
        self.project("attention","gated","self_attn.o_proj",self.d.hidden,rows(1),true)
    }
    fn latent(&mut self, a: LatentDescription) -> Result<()> {
        let i=a.indexer;
        self.plan.topology.interval(&[a.heads],rows(1))?;
        let ctx=self.plan.context; let pools=ctx/i.pool;
        let slots=add(i.token_budget,if i.select_tail { i.pool-1 } else { 0 })?;
        self.plan.inputs.extend([Buffer::new("latent_cache",&[ctx-1,a.kv_rank]),Buffer::new("index_keys",&[ctx-1,i.dim]),Buffer::new("index_gates",&[ctx-1,i.dim])]);
        let mut local=Buffer::new("local_slots",&[slots]); local.element=Element::U32; self.plan.inputs.push(local);
        self.plan.returns.extend(["next_latent_cache".into(),"next_index_keys".into(),"next_index_gates".into()]);
        self.project("query_down","input_norm","self_attn.q_a_proj",a.query_rank,Split::Replicated,false)?;
        self.norm("query_latent","query_down","self_attn.q_a_layernorm",a.query_rank);
        self.project("query","query_latent","self_attn.q_b_proj",mul(a.heads,a.key_dim)?,rows(a.key_dim),false)?;
        self.project("kv_down","input_norm","self_attn.kv_a_proj_with_mqa",a.kv_rank,Split::Replicated,false)?;
        self.norm("kv_latent","kv_down","self_attn.kv_a_layernorm",a.kv_rank);
        self.op("cache_append",Kind::CacheAppend,&["latent_cache","kv_latent"],vec![Buffer::new("next_latent_cache",&[ctx,a.kv_rank])],vec![],
            target(MLA,&["flash_hybrid_copy"]),Split::Replicated,&[]);
        self.project("index_query","query_latent","self_attn.indexer.wq_b",mul(i.heads,i.dim)?,Split::Replicated,false)?;
        self.project("index_key_raw","input_norm","self_attn.indexer.wk",i.dim,Split::Replicated,false)?;
        let weights=vec![self.weight("self_attn.indexer.k_norm.weight",&[i.dim],false),self.weight("self_attn.indexer.k_norm.bias",&[i.dim],false)];
        self.op("index_key",Kind::KeyNorm,&["index_key_raw"],vec![Buffer::new("index_key",&[i.dim])],weights,target(INDEXER,&["pooled_key_norm"]),Split::Replicated,&[]);
        let gate=self.weight("self_attn.indexer.index_kpool_compress_gate",&[i.dim,self.d.hidden],false);
        let t=weight_target(&gate,self.plan.topology)?;
        self.op("index_gate",Kind::Project,&["input_norm"],vec![Buffer::new("index_gate",&[i.dim])],vec![gate],t,Split::Replicated,&[]);
        let ape=self.weight("self_attn.indexer.index_kpool_compress_ape",&[i.pool,i.dim],false);
        self.op("pool",Kind::Pool,&["index_keys","index_gates","index_key","index_gate"],vec![Buffer::new("next_index_keys",&[ctx,i.dim]),
            Buffer::new("next_index_gates",&[ctx,i.dim]),Buffer::new("pool_keys",&[pools,i.dim])],vec![ape],target(INDEXER,&["flash_hybrid_copy","pooled_keys"]),Split::Replicated,&[]);
        self.project("index_head_weights","input_norm","self_attn.indexer.weights_proj",i.heads,Split::Replicated,false)?;
        self.op("pool_scores",Kind::ScorePools,&["index_query","pool_keys","index_head_weights"],vec![Buffer::new("pool_scores",&[pools])],vec![],
            target(INDEXER,&["pooled_scores"]),Split::Replicated,&[]);
        let mut ids=Buffer::new("selected",&[slots]); ids.element=Element::U32;
        let mut count=Buffer::new("selected_count",&[1]); count.element=Element::U32;
        self.op("select",Kind::Select,&["pool_scores"],vec![ids,count],vec![],target(INDEXER,&["qsa_fp32_select"]),Split::Replicated,&[]);
        self.op("gather",Kind::Gather,&["next_latent_cache","selected","selected_count"],vec![Buffer::new("selected_latent",&[slots,a.kv_rank])],vec![],
            target(MLA,&["flash_hybrid_copy"]),Split::Replicated,&[]);
        let mut kv=self.weight("self_attn.kv_b_proj",&[mul(a.heads,add(a.key_dim,a.value_dim)?)?,a.kv_rank],true); kv.split=rows(a.key_dim+a.value_dim);
        let mut t=weight_target(&kv,self.plan.topology)?; t.entries.push("flash_hybrid_copy");
        self.op("kv_expand",Kind::ExpandKv,&["selected_latent","selected_count"],vec![Buffer::new("keys",&[slots,a.heads,a.key_dim]),Buffer::new("values",&[slots,a.heads,a.value_dim])],
            vec![kv],t,rows(1),&[]);
        self.plan.steps.last_mut().unwrap().output_splits=vec![Split::Axis {axis:1,unit:1};2];
        self.plan.steps.last_mut().unwrap().scratch.push(Buffer::new("expanded_kv",&[slots,mul(a.heads,add(a.key_dim,a.value_dim)?)?]));
        self.op("scores",Kind::Scores,&["query","keys","selected_count","local_slots"],vec![Buffer::new("scores",&[a.heads,slots])],vec![],target(MLA,&["qsa_fp32_attention_scores"]),rows(1),&[]);
        self.op("softmax",Kind::Softmax,&["scores","selected_count"],vec![Buffer::new("probabilities",&[a.heads,slots])],vec![],target(MLA,&["qsa_fp32_softmax"]),rows(1),&[]);
        self.op("values",Kind::Values,&["probabilities","values","selected_count","local_slots"],vec![Buffer::new("attended",&[mul(a.heads,a.value_dim)?])],vec![],target(MLA,&["qsa_fp32_values"]),rows(a.value_dim),&["attended"]);
        self.project("attention","attended","self_attn.o_proj",self.d.hidden,rows(1),true)
    }
}
fn projection_target(storage: WeightStorage, width: u64, cols: u64) -> Target {
    match storage {
        WeightStorage::Plain if cols%16==0 && width%4==0 => target(BASE,&["flash_bf16_input","flash_bf16_pack","flash_bf16_mma"]),
        WeightStorage::Mxfp8 if cols%32==0 => target(BASE,&["flash_next_mxfp8_m1"]),
        WeightStorage::Fp8Block { block } if block[0] > 0 && block[1] > 0 && cols % block[1] == 0 => target(FP8,&["fp8_block_weight_only"]),
        WeightStorage::Nvfp4 { .. } if cols % 16 == 0 => target(BASE,&["fp4_exact_project"]),
        _ => gap("projection storage or input width has no kernel binding"),
    }
}
fn weight_target(w: &Weight,topology: Topology) -> Result<Target> {
    let interval=topology.interval(&w.shape,w.split)?;
    let (start,width)=interval.map_or((0,w.shape[0]),|(_,start,width)| (start,width));
    if let WeightStorage::Fp8Block {block}=w.storage {
        if block[0]>0 && start%block[0]!=0 { return Ok(gap("FP8 scale block crosses a row partition")); }
    }
    Ok(projection_target(w.storage,width,w.shape[1]))
}
fn expert_target(weights: &[Weight], topology: Topology) -> Result<Target> {
    let first=weight_target(&weights[0],topology)?;
    for w in weights { let next=weight_target(w,topology)?;
        if next != first { return Ok(gap("selected expert storage needs separate projection dispatch")); } }
    Ok(first)
}
