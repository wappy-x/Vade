//! Pure logical ownership, tensor selections and budgets for a model shape.
mod error;
mod memory;
mod owners;
mod sums;
pub use error::SplitError;
use crate::shapes::*;
use std::ops::Range;
type Result<T> = std::result::Result<T, SplitError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitRequest {
    /// Spark count within the shared limits.
    pub ranks: u32,
    /// Usable bytes and caller-supplied workspace bounds, one per rank.
    pub memory: Vec<u64>,
    pub reserve: Vec<u64>,
    /// Cached tokens per sequence, and full-context sequences funded.
    pub context: u64,
    pub sequences: u64,
    /// Bytes per cached K/V/index element and recurrent/conv state element.
    pub cache_bytes: u64,
    pub state_bytes: u64,
    /// Retain draft tensors and their cache/state.
    pub draft: bool,
    /// Own decoder expert down projections by output row blocks.
    pub expert_rows: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owners {
    // Absent layer kinds have 0..0 owners. Channels and rows are logical units.
    pub query_heads: Range<u64>,
    pub kv_heads: Range<u64>,
    pub value_heads: Range<u64>,
    pub key_heads: Range<u64>,
    pub shared: Range<u64>,
    pub dense: Range<u64>,
    pub vocab: Range<u64>,
    pub lexical_rows: Range<u64>,
    pub output_rows: Option<Range<u64>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection { All, Rows(Vec<Range<u64>>), Columns(Range<u64>), Absent }
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reduction { Attention, Linear, Shared, Dense, Expert(u32) }
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Collective { Sum { elements: u64 }, Gather { elements: u64 } }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ledger { pub weights: u64, pub cache: u64, pub state: u64, pub reserve: u64 }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitPlan {
    pub ranks: u32,
    pub owners: Vec<Owners>,
    pub ledgers: Vec<Ledger>,
    /// Decoder-layer collectives in execution order.
    pub layers: Vec<Vec<Collective>>,
    facts: Facts,
    draft: bool,
    expert_rows: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Facts {
    attention: Option<AttentionShape>,
    linear: Option<LinearShape>,
    expert: Option<ExpertShape>,
    dense: Option<u64>,
    lexical: Option<LexicalShape>,
}
impl SplitPlan {
    pub fn draft_enabled(&self) -> bool { self.draft }
    pub fn new(shape: &ModelShape, request: &SplitRequest) -> Result<Self> {
        let ranks = request.ranks;
        if !(MIN_SPARKS..=MAX_SPARKS).contains(&ranks) { return Err(SplitError::Ranks { ranks }); }
        for given in [request.memory.len(), request.reserve.len()] {
            if u32::try_from(given).ok() != Some(ranks) { return Err(SplitError::Budgets { ranks, given }); }
        }
        let facts = owners::facts(shape, request.expert_rows)?;
        let mut plan = Self {
            ranks,
            owners: (0..ranks).map(|r| owners::owners(shape, &facts, r, request)).collect::<Result<_>>()?,
            ledgers: Vec::new(),
            layers: sums::collectives(shape, request)?,
            facts,
            draft: request.draft,
            expert_rows: request.expert_rows,
        };
        plan.ledgers = memory::ledgers(&plan, shape, request)?;
        Ok(plan)
    }
    /// Contiguous ownership in FP8_BLOCK units, with expert-rotated remainders.
    pub fn expert_blocks(&self, rank: u32, expert: u32) -> Result<Range<u64>> {
        owners::expert_blocks(&self.facts, rank, self.ranks, expert)
    }
    /// Select original tensor coordinates from the role; the name is never parsed.
    pub fn selection(&self, rank: u32, tensor: &TensorEntry) -> Result<Selection> {
        owners::selection(self, rank, tensor)
    }
    /// One ascending K range per rank, retaining empty ranks in the fold order.
    pub fn segments(&self, reduction: Reduction) -> Result<Vec<Range<u64>>> {
        sums::segments(self, reduction)
    }
    fn owner(&self, rank: u32) -> Result<&Owners> {
        self.owners.get(index(rank)?).ok_or(SplitError::Shape { what: "rank index" })
    }
}
fn index(rank: u32) -> Result<usize> {
    usize::try_from(rank).map_err(|_| SplitError::Overflow { what: "rank index" })
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(SplitError::Overflow { what: "logical sum" })
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(SplitError::Overflow { what: "logical product" })
}
fn length(r: &Range<u64>) -> Result<u64> {
    r.end.checked_sub(r.start).ok_or(SplitError::Shape { what: "reversed range" })
}
fn scaled(r: &Range<u64>, unit: u64) -> Result<Range<u64>> {
    Ok(mul(r.start, unit)?..mul(r.end, unit)?)
}
fn required<T: Copy>(kind: Option<T>) -> Result<T> {
    kind.ok_or(SplitError::Shape { what: "tensor or reduction kind absent" })
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    use crate::{flash::{self, plan::{RankPlan, Selection as Old}, WeightFormat}, safetensors::Dtype};
    fn shape(model: u32) -> ModelShape {
        let (hidden, count, q, v, experts, selected, width) = match model { 0 => (2560,48,24,48,512,10,640), 1 => (2048,40,16,32,256,8,512), _ => (3072,48,32,64,256,8,1024) };
        let a = AttentionShape { query_heads:q, kv_heads:2, head_dim:256, rotary_dim:64, gated:true, index: (model == 0).then_some(IndexShape { heads:4, kv_heads:1, dim:128, budget:2048, compression:1 }) };
        let l = LinearShape { key_heads:16, value_heads:v, key_dim:128, value_dim:128, conv:4 };
        let ffn = Ffn::Experts(ExpertShape { count:experts, selected, width, shared_width:width, storage:ExpertStorage::PerExpert });
        ModelShape { hidden, vocab:248320, max_positions:131072, layers:(0..count).map(|i| LayerShape { mixer:if i%4==3 { Mixer::Attention(a) } else { Mixer::Linear(l) }, ffn }).collect(), draft:if model==0 { vec![LayerShape { mixer:Mixer::Attention(a), ffn }] } else { vec![] }, residual:if model==0 { Residual::Streams { count:4, rank:320 } } else { Residual::Single }, lexical:(model==0).then_some(LexicalShape { layer:1, parts:128, part_rows:2500012, row_dim:160 }), tensors:vec![] }
    }
    fn request(ranks: u32) -> SplitRequest { SplitRequest { ranks, memory:vec![u64::MAX; usize::try_from(ranks).unwrap()], reserve:vec![0; usize::try_from(ranks).unwrap()], context:0, sequences:0, cache_bytes:2, state_bytes:4, draft:false, expert_rows:false } }
    fn tensor(name: String, place: Place, part: Part, store: Store) -> TensorEntry { TensorEntry { name, dtype:Dtype::Bf16, shape:vec![1000,1000], role:Role { place, part, store } } }
    fn old(s: Selection) -> Old { match s { Selection::All=>Old::All, Selection::Rows(r)=>Old::Rows(r), Selection::Columns(r)=>Old::Columns(r), Selection::Absent=>Old::Absent } }
    #[test]
    fn flash_equivalence() {
        let mut entries = vec![tensor("lm_head.weight".into(),Place::Model,Part::Head,Store::Weight), tensor("model.language_model.embed_tokens.weight".into(),Place::Model,Part::Embedding,Store::Weight)];
        for (suffix, part) in [("q_proj",Part::Query),("k_proj",Part::Key),("v_proj",Part::Value),("o_proj",Part::Output),("q_norm",Part::QueryNorm),("k_norm",Part::KeyNorm),("indexer.index_qk_proj",Part::Index),("indexer.q_layernorm",Part::IndexNorm)] { entries.push(tensor(format!("model.language_model.layers.3.self_attn.{suffix}.weight"),Place::Layer(3),part,Store::Weight)); }
        for (suffix, part) in [("in_proj_qkv.weight",Part::LinearQkv),("conv1d.weight",Part::LinearConv),("in_proj_z.weight",Part::LinearZ),("out_proj.weight",Part::LinearOut),("in_proj_a.weight",Part::LinearHeads),("norm.weight",Part::LinearNorm),("A_log",Part::LinearHeads)] { entries.push(tensor(format!("model.language_model.layers.0.linear_attn.{suffix}"),Place::Layer(0),part,Store::Weight)); }
        for e in [0,1,7] { for (p,proj) in [("gate",Proj::Gate),("down",Proj::Down)] { for (s,store) in [("weight",Store::Weight),("weight_scale_inv",Store::BlockScale)] { entries.push(tensor(format!("model.language_model.layers.0.mlp.experts.{e}.{p}_proj.{s}"),Place::Layer(0),Part::Expert { id:Some(e),proj },store)); } } }
        entries.push(tensor("mtp.layers.0.mlp.experts.1.down_proj.weight".into(),Place::Draft { layer:Some(0) },Part::Expert { id:Some(1),proj:Proj::Down },Store::Weight));
        for (p,proj) in [("gate",Proj::Gate),("down",Proj::Down)] { entries.push(tensor(format!("model.language_model.layers.0.mlp.shared_expert.{p}_proj.weight"),Place::Layer(0),Part::Shared(proj),Store::Weight)); }
        for part in [0,64,127] { entries.push(tensor(format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{part}.weight"),Place::Layer(1),Part::Lexical { part },Store::Weight)); }
        for (name,place,part) in [("model.language_model.layers.0.mlp.gate.weight",Place::Layer(0),Part::Whole),("model.visual.blocks.0.attn.qkv.weight",Place::Unused,Part::Whole),("mtp.layers.0.self_attn.q_proj.weight",Place::Draft { layer:Some(0) },Part::Query)] { entries.push(tensor(name.into(),place,part,Store::Weight)); }
        for ranks in MIN_SPARKS..=MAX_SPARKS { for expert_rows in [false,true] { for draft in [false,true] {
            let q = SplitRequest { expert_rows,draft,..request(ranks) }; let plan = SplitPlan::new(&shape(0),&q).unwrap();
            let sum=Collective::Sum { elements:2560 }; let expected=if ranks==MIN_SPARKS { vec![] } else if expert_rows { vec![sum.clone(),sum,Collective::Gather { elements:6400 },Collective::Gather { elements:2560 }] } else { vec![sum,Collective::Sum { elements:28160 }] }; assert_eq!(plan.layers,vec![expected;48]);
            for rank in 0..ranks {
                let mut prior = RankPlan::new(u64::from(rank),u64::from(ranks)).unwrap(); prior.mtp=draft;
                prior.expert_output_rows=expert_rows.then(|| { let r=flash::partition(20,u64::from(rank),u64::from(ranks)).unwrap(); r.start*128..r.end*128 });
                let o=plan.owner(rank).unwrap();
                assert_eq!((&o.query_heads,&o.kv_heads,&o.value_heads,&o.key_heads,&o.shared,&o.vocab,&o.lexical_rows,&o.output_rows),(&prior.query_heads,&prior.kv_heads,&prior.value_heads,&prior.key_heads,&prior.shared_channels,&prior.vocabulary,&prior.ple_rows,&prior.expert_output_rows)); assert_eq!(o.dense,0..0);
                for e in 0..512 { assert_eq!(plan.expert_blocks(rank,e).unwrap(),prior.expert_blocks(u64::from(e))); }
                for t in &entries { assert_eq!(old(plan.selection(rank,t).unwrap()),prior.selection_for(&t.name,WeightFormat::Fp8).unwrap(),"{ranks}/{rank}: {}",t.name); }
            }
        } } }
    }
    #[test]
    fn ordered_segments() {
        for model in 0..3 { let mut s=shape(model); let e=match s.layers.first().unwrap().ffn { Ffn::Experts(e)=>e,_=>unreachable!() };
            s.layers.push(LayerShape { mixer:s.layers.first().unwrap().mixer,ffn:Ffn::Dense { width:e.shared_width } });
            for ranks in MIN_SPARKS..=MAX_SPARKS { let q=request(ranks); let p=SplitPlan::new(&s,&q).unwrap(); assert_eq!(p,SplitPlan::new(&s,&q).unwrap());
                let a=required(p.facts.attention).unwrap(); let l=required(p.facts.linear).unwrap();
                let mut reductions=vec![(Reduction::Attention,a.query_heads*a.head_dim),(Reduction::Linear,l.value_heads*l.value_dim),(Reduction::Shared,e.shared_width),(Reduction::Dense,e.shared_width)];
                reductions.extend((0..u32::try_from(e.count).unwrap()).map(|id| (Reduction::Expert(id),e.width)));
                for (r,k) in reductions { let ranges=p.segments(r).unwrap(); assert_eq!(u32::try_from(ranges.len()).unwrap(),ranks); let mut end=0; for range in ranges { assert_eq!(range.start,end); assert!(range.end>=range.start); end=range.end; } assert_eq!(end,k); }
            }
        }
    }
    #[test]
    fn errors_and_exact_budget() {
        let mut s=shape(0); for ranks in [0,9] { assert_eq!(SplitPlan::new(&s,&request(ranks)),Err(SplitError::Ranks { ranks })); }
        let mut q=request(MIN_SPARKS); q.memory.clear(); assert_eq!(SplitPlan::new(&s,&q),Err(SplitError::Budgets { ranks:MIN_SPARKS,given:0 }));
        s.tensors.push(tensor("whole".into(),Place::Model,Part::Whole,Store::Weight)); q=request(MIN_SPARKS); q.memory=vec![1_999_999];
        assert_eq!(SplitPlan::new(&s,&q),Err(SplitError::NoFit { rank:0,needed:2_000_000,available:1_999_999 })); q.memory=vec![2_000_000]; assert!(SplitPlan::new(&s,&q).is_ok());
        let e = SplitError::NoFit { rank:0, needed:2_000_000, available:1_999_999 }.to_string(); for text in ["Requested size does not fit.", "weights/rank 0", "2000000 bytes", "1999999 bytes", "Gap: 0.00 GiB (1 bytes)", "more Sparks"] { assert!(e.contains(text), "{e}"); }
        s.tensors=vec![tensor("stacked".into(),Place::Layer(0),Part::Expert { id:None,proj:Proj::Gate },Store::Weight)]; assert!(matches!(SplitPlan::new(&s,&q),Err(SplitError::Unsupported { .. })));
    }
    #[test]
    fn storage_and_absent_kinds() {
        let s=shape(0); let p=SplitPlan::new(&s,&request(MAX_SPARKS)).unwrap();
        for part in [Part::Embedding,Part::Head,Part::Shared(Proj::Gate),Part::Dense(Proj::Down),Part::Lexical { part:0 },Part::Query,Part::Key,Part::Value,Part::Output,Part::QueryNorm,Part::KeyNorm,Part::Index,Part::IndexNorm,Part::LinearQkv,Part::LinearConv,Part::LinearZ,Part::LinearOut,Part::LinearHeads,Part::LinearNorm] { assert!(matches!(p.selection(0,&tensor("scale".into(),Place::Layer(0),part,Store::Scalar)),Err(SplitError::Unsupported { .. }))); }
        for part in [Part::Lexical { part:0 },Part::QueryNorm,Part::KeyNorm,Part::IndexNorm,Part::LinearConv,Part::LinearNorm] { assert!(matches!(p.selection(0,&tensor("scale".into(),Place::Layer(0),part,Store::BlockScale)),Err(SplitError::Unsupported { .. }))); }
        for store in [Store::BlockScale,Store::Scalar] { assert_eq!(p.selection(0,&tensor("whole".into(),Place::Model,Part::Whole,store)).unwrap(),Selection::All); }
        let t=tensor("expert scalar".into(),Place::Layer(0),Part::Expert { id:Some(0),proj:Proj::Gate },Store::Scalar); assert_eq!(p.selection(0,&t).unwrap(),Selection::All); assert_eq!(p.selection(MAX_SPARKS-1,&t).unwrap(),Selection::Absent);
        for place in [Place::Unused,Place::Draft { layer:Some(0) }] { assert_eq!(p.selection(0,&tensor("disabled".into(),place,Part::Query,Store::BlockScale)).unwrap(),Selection::Absent); }
        assert!(matches!(p.selection(0,&tensor("dense".into(),Place::Layer(0),Part::Dense(Proj::Gate),Store::Weight)),Err(SplitError::Shape { .. })));
        for dense in [false,true] { let mut s=shape(0); for l in s.layers.iter_mut().chain(&mut s.draft) { l.ffn=if dense { Ffn::Dense { width:640 } } else { match l.ffn { Ffn::Experts(mut e)=>{ e.shared_width=0; Ffn::Experts(e) },f=>f } }; }
            let p=SplitPlan::new(&s,&request(MAX_SPARKS)).unwrap(); assert!(matches!(p.segments(Reduction::Shared),Err(SplitError::Shape { .. })));
            for proj in [Proj::Gate,Proj::Up,Proj::Down] { assert!(matches!(p.selection(0,&tensor("shared".into(),Place::Layer(0),Part::Shared(proj),Store::Weight)),Err(SplitError::Shape { .. }))); }
        }
        for case in 0..6 { let mut s=shape(0); match case { 0=>s.hidden=0,1=>s.vocab=0,2=>s.layers.first_mut().unwrap().ffn=Ffn::Dense { width:0 },3|4=>{ for l in s.layers.iter_mut().chain(&mut s.draft) { if let Ffn::Experts(ref mut e)=l.ffn { if case==3 { e.width=0; } else { e.selected=0; } } } },_=>s.lexical.as_mut().unwrap().part_rows=0 }; assert!(matches!(SplitPlan::new(&s,&request(MIN_SPARKS)),Err(SplitError::Shape { .. }))); }
    }
    #[test]
    fn resident_ledgers() {
        let mut s=shape(0); let mut gate=tensor("shared gate".into(),Place::Layer(0),Part::Shared(Proj::Gate),Store::Weight); gate.shape=vec![640,3];
        let mut down=tensor("shared down".into(),Place::Layer(0),Part::Shared(Proj::Down),Store::Weight); down.shape=vec![3,640];
        let mut qkv=tensor("qkv".into(),Place::Layer(0),Part::LinearQkv,Store::Weight); qkv.shape=vec![10240,2,3]; s.tensors=vec![gate,down,qkv];
        let mut q=SplitRequest { context:3,sequences:2,draft:true,reserve:vec![11,13,17],..request(3) };
        let expected=vec![Ledger { weights:45600,cache:99840,state:79626240,reserve:11 },Ledger { weights:45600,cache:179712,state:79626240,reserve:13 },Ledger { weights:45504,cache:99840,state:79626240,reserve:17 }];
        q.memory=vec![79771691,79851565,79771601]; let p=SplitPlan::new(&s,&q).unwrap(); assert_eq!(p.ledgers,expected);
        for rank in 0..q.ranks { let mut low=q.clone(); let available=low.memory.get_mut(usize::try_from(rank).unwrap()).unwrap(); let needed=*available; *available-=1; assert_eq!(SplitPlan::new(&s,&low),Err(SplitError::NoFit { rank,needed,available:needed-1 })); }
        q.draft=false; let p=SplitPlan::new(&s,&q).unwrap(); assert_eq!(p.ledgers.iter().map(|l| l.cache).collect::<Vec<_>>(),vec![92160,165888,92160]); assert!(p.ledgers.iter().all(|l| l.state==79626240));
        q.context=u64::MAX; assert!(matches!(SplitPlan::new(&s,&q),Err(SplitError::Overflow { .. })));
        let mut bad=s.clone(); bad.tensors.first_mut().unwrap().shape=vec![1,3]; assert!(matches!(SplitPlan::new(&bad,&request(3)),Err(SplitError::Shape { .. })));
        bad=s; bad.tensors.get_mut(1).unwrap().shape=vec![3,1]; assert!(matches!(SplitPlan::new(&bad,&request(3)),Err(SplitError::Shape { .. })));
    }
}
