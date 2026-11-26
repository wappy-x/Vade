//! Bounded causal recipe interpreter. Full service admission stays separate.
#![allow(clippy::chunks_exact_to_as_chunks, clippy::collapsible_if, clippy::needless_borrow, clippy::needless_late_init, reason = "kept as on main while lane-byte-cut edits this file; fix after it merges")]
use crate::{Result,causal_layer,mxfp8_projection::Mxfp8Projection,layer_driver::LayerBoundary};
use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,LayerPlan,LayerSchedule,Topology},state::StatePlan,storage::WeightStorage,description::TokenMixer},safetensors::Dtype};
use spark_gpu_probe::{causal_ops::{Shape,Op},norm_boundary::{NormShape,WeightType}};
use std::{cell::RefCell,collections::BTreeMap,rc::Rc};
#[path="bounded_decoder/parameters.rs"] mod parameters;
// Tests build it on every host; outside a Linux `kernel-bench` build only its tests use it.
#[cfg(any(test,all(target_os="linux",feature="kernel-bench")))]
#[cfg_attr(not(all(target_os="linux",feature="kernel-bench")),allow(dead_code))]
#[path="bounded_decoder/memo_proof.rs"] mod memo_proof;
pub use parameters::{StaticParameters,ParameterCounts};
use parameters::StaticParameterPlan;
pub(crate) mod owners;
pub const PROMPT_ROWS:usize=4;
pub const MEMO_SLAB_ROWS:usize=64;
/// Larger prompt groups copy their recurrent state prefix out of the row output. Every host reads it: the start
/// line's planned charge of a kept prompt (`row_split::Factory::kept_prompt_bytes`) counts what a shorter group pins.
pub(crate) const STATE_VIEW_ROWS:usize=256;
#[cfg(any(target_os="linux",test))]
pub(crate) fn copy_prompt_state(tokens:usize)->bool {tokens>STATE_VIEW_ROWS}
fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
/// The residual add of a BF16 model on the host, `round_bf16(x + y)` a word: both read from `c`, the sum uploaded to it.
fn host_bf16_sum<C:Compute+?Sized>(c:&C,x:&C::Value,y:&C::Value)->Result<C::Value> {
    let (x,y)=(c.read(x)?,c.read(y)?);c.upload(&bf16_sums(&x,&y)?)
}
/// The host form's words, or its refusal: the error of the first word whose sum is not finite or rounds past BF16.
fn bf16_sums(x:&[f32],y:&[f32])->Result<Vec<f32>> {
    need(x.len()==y.len(),"residual sum extent differs")?;
    x.iter().zip(y).map(|(a,b)|engine_model::plan::layer::cast::round_bf16(a+b).map_err(|e|e.to_string())).collect()
}
pub(crate) fn greedy_logits(logits:&[f32],words:usize)->Result<u32> {
    need(!logits.is_empty() && logits.len()==words && logits.iter().all(|value|value.is_finite()),"bounded logits extent or value differs")?;
    let mut best=(logits[0],0);for (id,&value) in logits.iter().enumerate().skip(1) {
        best=spark_gpu_probe::next_math::better(best,(value,u32::try_from(id).map_err(|_|"bounded logits ID overflow")?));
    }Ok(best.1)
}
/// DRAFT CONFIDENCE: `greedy_logits` and the probability of its pick under the softmax of `logits`.
pub(crate) fn greedy_logits_mass(logits:&[f32],words:usize)->Result<(u32,Option<f32>)> {
    let pick=greedy_logits(logits,words)?;let best=f64::from(logits[pick as usize]);
    let mass=logits.iter().map(|&value|(f64::from(value)-best).exp()).sum::<f64>();Ok((pick,Some((1.0/mass) as f32)))
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
mod asked;
#[cfg_attr(not(target_os="linux"),allow(dead_code))]
mod memo_batches;
mod residual_plan;
mod initial_state;
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
/// The fold step (ExpertSum) of the routed down step `index`, when that fold is the only reader of the down outputs.
fn fold_of(index:usize,plan:&LayerPlan)->Option<(usize,&engine_model::plan::layer::Step)> {
    let down=&plan.steps[index];
    if down.kind!=Kind::ExpertDown || down.outputs.len()!=1 || down.inputs.len()!=2 {return None;}
    let name=&down.outputs[0].name;
    let mut readers=plan.steps.iter().enumerate().filter(|(_,step)|step.inputs.iter().any(|input|input==name));
    let (Some((at,fold)),None)=(readers.next(),readers.next()) else {return None;};
    if at<=index || fold.kind!=Kind::ExpertSum || fold.outputs.len()!=1 || !(3..=4).contains(&fold.inputs.len()) || fold.inputs[2]!=*name
        || fold.inputs[0]!=down.inputs[1] || !fold.weights.is_empty() || plan.returns.iter().any(|kept|kept==name) {return None;}
    Some((at,fold))
}
/// Rows that a selection keeps at this context: every row of the kept pools, then the tail rows after the last
/// complete pool. It is the count word of the selection output for finite and infinite scores (a NaN score writes -2).
pub fn selected_count(pool:u64,budget:u64,tail:bool,context:u64)->u64 {
    if pool==0 {return 0;}let groups=context/pool;groups.min(budget/pool)*pool+if tail {context-groups*pool} else {0}
}
/// The decode step's Select row of a latent layer at `context`, as layer_group builds it, has no host-known form: the
/// step waits for the device's selection (select_admission::known_selection gives None). True on any shape it cannot form.
pub(crate) fn selection_read(i:&engine_model::plan::description::PoolIndexerDescription,context:u64)->bool {
    let Some(mut row)=spark_gpu_probe::prompt_ops::Shape::new(spark_gpu_probe::prompt_ops::Op::SelectRows,1) else {return true;};
    let slots=i.token_budget.checked_add(if i.select_tail {i.pool.saturating_sub(1)} else {0});
    let (Some(groups),Ok(dim),Ok(taps),Ok(first))=(context.checked_div(i.pool),u32::try_from(i.pool),u32::try_from(i.token_budget),u32::try_from(context)) else {return true;};
    let (Ok(rows),Some(Ok(slots)))=(u32::try_from(groups),slots.map(u32::try_from)) else {return true;};
    row.dim=dim;row.taps=taps;row.rows=rows;row.cols=u32::from(i.select_tail);row.first=first;row.slots=slots;
    spark_gpu_probe::select_admission::known_selection(row).is_none()
}
/// One chat's Gather of an absorbed layer, as `gather_absorbed` takes it: its shape, its inputs, and the count that the
/// selection rule gives for its context.
pub type ChatGather<'a,V>=(Shape,[Option<&'a V>;4],f32);
/// The recurrent layer outputs and the operands needed for prefix acceptance.
pub struct RecurrentRows<V> {pub history:V,pub state:V,pub gated:V,pub saved:Option<(V,V)>}
/// A value set for one call: `set(new)` at once, and `set(old)` when the guard drops, on every path (a good return,
/// an error return, a panic). The device sets its row pass flag for one launch with it (`Compute::dot_rows`).
/// Built where a user of it is: the device (Linux) and the tests.
#[cfg(any(target_os="linux",test))]
pub(crate) struct Restore<T:Copy,F:FnMut(T)> {old:T,set:F}
#[cfg(any(target_os="linux",test))]
impl<T:Copy,F:FnMut(T)> Restore<T,F> {pub(crate) fn new(old:T,new:T,mut set:F)->Self {set(new);Self {old,set}}}
#[cfg(any(target_os="linux",test))]
impl<T:Copy,F:FnMut(T)> Drop for Restore<T,F> {fn drop(&mut self) {(self.set)(self.old)}}
#[cfg(test)]
mod restore_tests {
    use super::{Restore,Result};
    use std::cell::Cell;
    /// The flag has its old value after the call on the good path, after a refused launch, and after a panic.
    #[test]
    fn a_flag_set_for_one_call_has_its_old_value_after_it_on_every_path() {
        for old in [false,true] {
            let (flag,sets)=(Cell::new(old),Cell::new(0));
            let call=|refuse:bool|->Result<u32> {
                let guard=Restore::new(flag.get(),true,|on:bool|{flag.set(on);sets.set(sets.get()+1);});
                assert!(flag.get());
                let launch:Result<u32>=if refuse {Err("the launch is refused".into())} else {Ok(7)};
                drop(guard);launch
            };
            assert_eq!(call(false),Ok(7));assert_eq!(flag.get(),old);
            assert!(call(true).is_err());assert_eq!(flag.get(),old);
            let early=||->Result<u32> {let _guard=Restore::new(flag.get(),true,|on:bool|flag.set(on));Err("refused before the end".into())};
            assert!(early().is_err());assert_eq!(flag.get(),old);
            let panicked=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{let _guard=Restore::new(flag.get(),true,|on:bool|flag.set(on));panic!("a launch panics")}));
            assert!(panicked.is_err());assert_eq!(flag.get(),old);
            assert_eq!(sets.get(),4);
        }
    }
}
/// The shape of one launch of arm `RowDots` for `Compute::dot_rows`: `words` query words of rows of `width`, `count`
/// stack rows.
pub fn row_dots_shape(words:usize,count:usize,width:usize)->Result<Shape> {
    let number=|n:usize|u32::try_from(n).map_err(|_|"row dots exceed the operation ABI".to_owned());
    if width==0 || words==0 || !words.is_multiple_of(width) || count==0 {return Err("row dots need whole query rows and a stack row".into());}
    Ok(Shape {op:Op::RowDots,heads:number(count)?,dim:0,taps:0,rows:number(words/width)?,cols:number(width)?,epsilon:0.0,bound:0.0})
}
/// The operands of `Compute::window_one`: `rows` query rows of `heads` heads, row-major. `query` and `moved` are the
/// products before their clips (scales empty for none, `position` the factor of each row in a layer that has one);
/// `keys` and `values` the views of `view` rows that every row reads; `steps` the live count and the slot of each row.
pub struct WindowOne<'a,V> {
    pub rows:usize,pub heads:usize,pub kv_heads:usize,pub view:usize,pub extent:usize,pub scale:f32,
    pub query:&'a V,pub query_scale:&'a [f32],pub query_bound:f32,pub moved:&'a V,pub moved_scale:&'a [f32],pub moved_bound:f32,
    pub position:Option<&'a V>,pub keys:&'a V,pub values:&'a V,pub written:Option<(&'a V,&'a V)>,pub rel_table:&'a [f32],pub sink:&'a V,pub gate:&'a V,pub steps:&'a V,
}
/// Operands of `Compute::window_split`: GQA-packed Split-KV tensor-core decode/verify attention (M=1..16 single-chat
/// with optional zero-copy provisional sliding-window `delta`, or C=2..16 multi-chat `planes`).
pub struct WindowSplit<'a,V> {
    pub base:WindowOne<'a,V>,pub is_ring:bool,pub first_pos:u64,pub row_pos_live:&'a [(u64,usize)],
    pub delta:Option<(&'a V,&'a V)>,pub planes:&'a [(V,V)],
}
/// The operands of `Compute::conv_finish`: arm `ShortConv` (`shape`, no addend) on a key or value ring with
/// `product` as the input of the ring row in `step`, then `finish` (`HeadClip` at threshold `param`, or `RoundedNorm`
/// at epsilon `param`, with no scale or weight) on each head, and the finished row packed into words `plane_word ..`
/// of its plane.
/// Operands of `Compute::kv_finish_pair`: multi-row Key and Value `ShortConv` + `HeadClip`/`RoundedNorm` +
/// `PackBf16` + `every` in one launch.
pub struct KvFinishPair<'a,V> {
    pub chat_rings:&'a [(&'a V,&'a V,usize)],
    pub rows:usize, pub kv_heads:usize, pub taps:usize, pub first_slot:usize,
    pub key_bound:f32, pub val_eps:f32, pub key_scale:f32, pub val_scale:f32,
    pub key_in:&'a V, pub key_ring:&'a V, pub key_fixed:&'a [f32], pub key_made:Option<&'a V>,
    pub val_in:&'a V, pub val_ring:&'a V, pub val_fixed:&'a [f32], pub val_made:Option<&'a V>,
}
/// Operands of `Compute::route_fold_conv`: fused multi-row `RouteFold` + `AfterExperts` `ShortConv` + `every` +
/// `Residual` + `Round` in one launch.
pub struct RouteFoldConv<'a,V> {
    pub chat_rings:&'a [(&'a V,usize)],
    pub rows:usize, pub picks:usize, pub dim:usize, pub weight_stride:usize, pub weight_offset:usize,
    pub taps:usize, pub first_slot:usize, pub bound:f32, pub half:bool,
    pub route:&'a V, pub downs:&'a V, pub residual:&'a V, pub fixed:&'a [f32], pub ring:&'a V,
}
pub struct ConvFinish<'a,V> {
    pub shape:Shape,pub finish:Op,pub param:f32,pub fixed:&'a [f32],pub made:Option<&'a V>,pub step:&'a V,pub product:&'a V,pub plane_word:usize,
}
pub struct PrefillKv<'a, V> {
    pub shape: Shape, pub rows: usize, pub ring_first: usize, pub ring: &'a V, pub input: &'a V,
    pub made: Option<&'a V>, pub fixed: &'a [f32], pub write: spark_gpu_probe::fast_ops::KvWrite,
}
#[path="bounded_decoder/c8_fast.rs"] mod c8_fast;
pub use c8_fast::{KvDirectTarget,KvFinishDirect,ShortConvDirect};

pub trait Compute {
    /// Set the loop arithmetic phase and return the previous setting.
    fn loop_prefill(&self, _prefill: bool) -> bool { false }
    /// One ordered execution owns temporary buffers through state publication.
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {run()}
    fn resident_weights_ready(&self)->bool {false}
    fn memo_summary(&self,_:&CheckpointAdmission)->Option<crate::projection_memo::Summary> {None}
    fn freeze_memo(&self,plan:crate::projection_memo::Plan)->Result<crate::projection_memo::Plan> {
        if plan==crate::projection_memo::Plan::full() {Ok(plan)} else {Err("compute cannot enforce a projection memo bound".into())}
    }
    /// Diagnostic: graph pieces captured under a new key, captured again under a key seen before, evicted, and updated in
    /// place for new launch grids, since the load. A recorded piece is keyed by the grids of its launches; captures that grow
    /// with every step name a grid that moves and has no retained executable to take it.
    fn capture_counts(&self)->[u64;4] {[0;4]}
    fn fatal_error(&self)->Option<String> {None}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {None}
    /// Diagnostic: device vectors newly allocated so far, as (count, bytes). None off the device.
    fn vector_allocations(&self)->Option<(usize,usize)> {None}
    /// Diagnostic: (new count, new bytes, spares reused) so far. None off the device.
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {None}
    /// A small pass (a decode step or a verify pass of a few rows) starts: leave `room` bytes under the vector pool's
    /// cap. Free storage only, no value changes. Without it a full pool refuses every return, and each pass buys its
    /// working set anew (measured on the Inkling loop decoder, four-0: 1,072 new buffers a 4-row verify pass at 124
    /// tokens of context, 2,037 a plain step after a 2,008-token prompt). The body here does nothing.
    fn pass_room(&self,_room:usize) {}
    /// A service whose memory admission charges `bytes` for the values of one step of rows: the vector pool may keep that
    /// many spare bytes, so the next step (the next piece of a prompt, the next prompt) takes its buffers back instead
    /// of allocating them again (measured on the Inkling loop decoder: about 290 buffers freed and 280
    /// allocated between the two pieces of a cold 8K prompt, 54 ms with the GPU idle, and 39 ms of frees before the
    /// first token). Free storage only, no value changes. The body here does nothing.
    fn step_spares(&self,_bytes:usize) {}
    /// A small pass of `rows` rows of the loop decoder starts (a verify pass of drafted tokens, a step of a few chats),
    /// or, with `rows` 1, it ends. A backend that records launches may then record the row kernels of the pass (the
    /// gathers and the dots of the mix, `prompt_ops`) with the rest of its launches, as it does for the verify pass of the
    /// bounded decoder. Launch order only: no value changes. Without it each row kernel of such a pass is a direct
    /// launch that cuts the recorded launches before it into a graph of its own (measured on the Inkling loop
    /// decoder, four-0: 434 direct launches and 510 graph launches a 4-row verify pass, 9.5 ms of device idle in
    /// 103.4 ms). The body here does nothing.
    fn rows_small_pass(&self,_rows:usize) {}
    fn serving_trace(&self)->bool {false}
    fn profile_tick_begin(&self,_tick:u64) {}
    fn profile_tick_end(&self) {}
    /// Host work outside the device family timers, in diagnostic cost runs.
    fn plan_cost(&self,_name:&str,_ns:u128) {}
    fn trace_start(&self) {}
    fn cost_report(&self,_phase:&str) {}
    /// The fused mHC launches of row groups (chat steps, drafting check passes) this thread has queued. No default body:
    /// a backend that forgot it would read 0 in silence. A backend with no fused launches says 0 itself.
    fn mhc_row_launches(&self)->u64;
    type Value;type Pass;
    /// Host metadata only. List every allocation this state plane keeps alive.
    /// False means this backend cannot keep this plane without more work.
    fn state_allocations(&self,_:&Self::Value,_:&mut Vec<(usize,usize)>)->bool {false}
    /// Growing straight KV plane hooks (`plane_page_bytes == 0`: full planes up front; GPU VMM uses 2 MiB pages).
    fn plane_page_bytes(&self)->usize {0}
    fn plane_alloc(&self,maximum_words:usize,_initial_words:usize)->Result<Self::Value> {let none=self.zeros(0)?;self.state_restore(&none,maximum_words)}
    fn plane_grow(&self,_plane:&mut Self::Value,_needed_words:usize,_maximum_words:usize)->Result<()> {Ok(())}
    fn plane_fork(&self,source:&Self::Value,committed_words:usize,needed_words:usize,_maximum_words:usize)->Result<Self::Value> {
        let live=self.range(source,0,committed_words)?;let target=needed_words.max(committed_words);
        if committed_words<target {self.state_restore(&live,target)} else {self.rows_append(None,0,&live,1)}
    }
    fn plane_trim(&self,_plane:&mut Self::Value,_keep_words:usize)->Result<()> {Ok(())}
    fn plane_prefix_allocations(&self,v:&Self::Value,_committed_words:usize,out:&mut Vec<(usize,usize)>)->bool {self.state_allocations(v,out)}
    fn fund_kv_pages(&self,_bytes:usize) {}
    fn kv_reserve_cohort(&self,_chats:usize) {}
    fn kv_conversation_ended(&self,_bytes:usize) {}
    fn projects_complete_rows(&self)->bool {false}
    fn sparse_topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn owned_expansion(&self,_:&str,_:&Self::Value,_:usize,_:crate::sparse_heads::Layout)->Result<Option<Self::Value>> {Ok(None)}
    fn owned_expansion_rows(&self,_:&str,_:&Self::Value,_:usize,_:crate::sparse_heads::Layout)->Result<Option<Self::Value>> {Ok(None)}
    fn gather_attended(&self,value:&Self::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Self::Value> {
        need(layout.local_heads()==layout.total_heads(),"compute has no attended gather")?;
        self.range(value,0,tokens*layout.total_heads()*layout.value_dim())
    }
    fn prepare(&self,_:&CheckpointAdmission)->Result<()> {Ok(())}
    /// A session opens while `live` other sessions use this owner. No default: a wrapper that passed `prepare` on in
    /// its place would bring back the memo reset that takes the rows the other sessions decode with.
    fn prepare_beside(&self,c:&CheckpointAdmission,live:usize)->Result<()>;
    /// More resident matrices of the prepared checkpoint (the appended predictor layer).
    fn prepare_names(&self,_:&CheckpointAdmission,_:&[String])->Result<()> {Ok(())}
    fn projection(&self,_:&CheckpointAdmission,_:&str,_:&Self::Value,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    /// The greedy ID of each of `rows` equal rows of `logits`, picked where the values are: the highest value, the lowest ID on a
    /// tie, a logit that is not finite refused (the pick of `greedy_logits`). None: this backend has no such form and the caller
    /// reads the logits. No default body: a wrapper must hand this on.
    fn rows_greedy(&self,logits:&Self::Value,rows:usize)->Result<Option<Vec<u32>>>;
    /// Fused `Op::Softcap` (`cap(rn(x), eps, bound)`) + `rows_greedy` without writing capped logits; `None` falls back.
    fn rows_softcap_greedy(&self,_raw:&Self::Value,_rows:usize,_bound:f32,_eps:f32)->Result<Option<Vec<u32>>> {Ok(None)}
    /// The dots of each query row of `queries` (host words, `[queries][width]`) with each of the `count` rows of `stack`
    /// (`[count][width]`): `[queries][count]`, query then stack row. Each dot is one chain over the width in rising order
    /// from +0.0, each product rounded before its add. The body here is one launch of arm `RowDots` (`row_dots_shape`).
    /// A backend may run another launch with the same words.
    fn dot_rows(&self,queries:&[f32],stack:&Self::Value,count:usize,width:usize)->Result<Self::Value> {
        self.operation(row_dots_shape(queries.len(),count,width)?,[Some(stack),None,None,None],queries)
    }
    /// `rows` rows of one block arm in one launch: the rows entry of the arms (`spark_gpu_probe::block_rows`, kernel
    /// entry `resident_launch_block_rows`; Ruling R6g: Inkling's arms never run in an entry that other families
    /// launch). Row `r` runs `operation(s, ..)` on its own part of each input, or on the whole input where bit `n` of
    /// `shared` is set; the output holds the output of each row in turn. The counts are `block_rows::lengths`.
    /// None: this backend has no such launch, and the caller runs one launch a row (the same words).
    fn block_rows(&self,_:Shape,_:usize,_:u32,_:[Option<&Self::Value>;4],_:&[f32])->Result<Option<Self::Value>> {Ok(None)}
    /// One scores or values row per chat, each reading its own KV plane.
    fn block_planes(&self,_:Shape,_:[&Self::Value;3],_:&[Self::Value])->Result<Option<Self::Value>> {Ok(None)}
    /// `WindowScores` or `WindowValues` over a virtual chronological plane `(ring[start_slot..], delta)` with zero copy.
    #[allow(clippy::too_many_arguments)]
    fn block_ring_delta(&self,_:Shape,_:usize,_:u32,_:[&Self::Value;3],_:&Self::Value,_:&Self::Value,_:usize,_:usize,_:usize)->Result<Option<Self::Value>> {Ok(None)}
    /// List-form `MixBlend` writing directly in transposed `[asked][rows][dim]` layout (`FAST_MIX_BLEND`),
    /// eliminating post-mix `PermuteRows` swaps. Returns `Ok(None)` to fall back to `block_rows` + `swap`.
    #[allow(clippy::too_many_arguments)]
    fn mix_blend_transposed(&self, _rows: usize, _asked: usize, _kept: usize, _dim: usize, _scale: f32, _scores: &Self::Value, _joined: &Self::Value) -> Result<Option<Self::Value>> { Ok(None) }
    fn mix_prefill(&self, _: usize, _: usize, _: f32, _: bool, _: &[f32], _: &[&Self::Value]) -> Result<Option<Self::Value>> { Ok(None) }
    /// Fused decode mix (`rows = 1..=16`): plain norm of `stream`, sequential dot chains of `queries` with `list`
    /// and the normed stream, softmax, and blend in one launch. Returns `Ok(None)` to fall back to separate ops.
    #[allow(clippy::too_many_arguments)]
    fn mix_rows(&self,rows:usize,dim:usize,scale:f32,eps:f32,_transpose:bool,keep_normed:bool,
        queries:&[f32],list:&[&Self::Value],stream:&Self::Value)->Result<Option<(Option<Self::Value>,Self::Value)>> {
        if rows!=1 {return Ok(None);}
        Ok(self.mix_one(queries,list,stream,dim,eps,scale)?.map(|(n,m)|(keep_normed.then_some(n),m)))
    }
    fn mix_one(&self,_queries:&[f32],_list:&[&Self::Value],_x:&Self::Value,_width:usize,_epsilon:f32,_scale:f32)->Result<Option<(Self::Value,Self::Value)>> {Ok(None)}
    /// Fused decode mix (`MIX_ONE`) with post-blend `RoundedNorm`. When `keep_mixed` is false, skips writing un-normed blend.
    #[allow(clippy::too_many_arguments)]
    fn mix_rows_normed(&self,_:usize,_:usize,_:f32,_:f32,_:f32,_:bool,_:bool,_:bool,_:&[f32],_:&[&Self::Value],_:&Self::Value,_:Option<&[f32]>,_:bool)->Result<Option<(Option<Self::Value>,Option<Self::Value>,Self::Value)>> {Ok(None)}
    /// Fused `GateBlend` + weighted `RoundedNorm` for 1..=3 gates in one launch.
    fn gate_blend_norm(&self,_:usize,_:usize,_:f32,_:&[(&Self::Value,&Self::Value,&Self::Value,Option<&[f32]>)])->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// Strided `RouteFold`: reads weights for row `r` at `route[r * weight_stride + weight_offset .. + picks]`.
    #[allow(clippy::too_many_arguments)]
    fn route_fold_strided(&self,_:usize,_:usize,_:usize,_:usize,_:usize,_:&Self::Value,_:&Self::Value)->Result<Option<Self::Value>> {Ok(None)}
    /// `mix_prefill` with `transpose` and the gate norm of every blended row in one launch: (the mix `[asked][rows][dim]`,
    /// its rounded norm with no weight and epsilon `eps`, each row normed alone). Each word is the word of `mix_prefill`
    /// and of `operation` with arm `RoundedNorm` on it. None: no such launch here; the caller runs the two.
    fn mix_prefill_normed(&self, _: usize, _: usize, _: f32, _: f32, _: &[f32], _: &[&Self::Value]) -> Result<Option<(Self::Value, Self::Value)>> { Ok(None) }
    /// Read a single chat's convolution directly from its old ring and the new rows.
    fn short_conv_prefill(&self, _:Shape, _:usize, _:usize, _:[Option<&Self::Value>;4], _:&[f32]) -> Result<Option<Self::Value>> { Ok(None) }
    /// Convolve, clip or norm, and pack a prompt tile directly into its KV plane in one launch.
    /// Returns false before any writes when the operands do not fit the launch.
    fn prefill_kv(&self, _: &PrefillKv<'_, Self::Value>, _: &mut Self::Value) -> Result<bool> { Ok(false) }
    /// Arm `ShortConv` directly from `inputs` and `ring` (`[inputs, token, addend, ring]`), writing `every`; `None` falls back.
    fn short_conv_keep(&self, _:Shape, _:usize, _:spark_gpu_probe::fast_ops::ShortConvRing, _:[Option<&Self::Value>;4], _:&[f32]) -> Result<Option<(Self::Value,Self::Value)>> { Ok(None) }
    /// Direct multi-chat ring-pointer `short_conv_keep` (`[inputs, token, addend]`, per-chat `rings`), avoiding `join` copies.
    fn short_conv_keep_multi(&self, _:Shape, _:usize, _:spark_gpu_probe::fast_ops::ShortConvMulti, _:[Option<&Self::Value>;3], _:&[&Self::Value], _:&[f32]) -> Result<Option<(Self::Value,Self::Value)>> { Ok(None) }
    fn kv_finish_pair(&self, _:&KvFinishPair<'_,Self::Value>) -> Result<Option<[Self::Value;4]>> { Ok(None) }
    fn kv_finish_direct(&self, _:&KvFinishDirect<'_,Self::Value>, _:&mut [KvDirectTarget<'_,Self::Value>]) -> Result<Option<[Self::Value;2]>> { Ok(None) }
    fn short_conv_direct(&self, _:&ShortConvDirect<'_,Self::Value>) -> Result<Option<(Self::Value,Self::Value)>> { Ok(None) }
    fn route_fold_conv(&self, _:&RouteFoldConv<'_,Self::Value>) -> Result<Option<(Self::Value,Self::Value)>> { Ok(None) }
    fn router_pick_compact(&self, _:Shape, _:usize, _:&Self::Value, _:&[f32]) -> Result<Option<(Self::Value,Self::Value)>> { Ok(None) }
    /// SM121a BF16 tensor-core fused causal/sliding-window attention (`FLASH_WINDOW_ATTN`, kind 101 in `kernels-inkling`).
    /// Replaces `RelBias + WindowScores + SinkSoftmax + WindowValues` for contiguous chat spans when supported;
    /// returns `Ok(None)` to fall back to the materialized path.
    #[allow(clippy::too_many_arguments)]
    fn flash_window_attn(
        &self, _rows: usize, _heads: usize, _kv_heads: usize, _head_dim: usize, _rel_dim: usize, _view: usize, _extent: usize,
        _scale: f32, _first_position: u64, _query: &Self::Value, _keys: &Self::Value, _values: &Self::Value, _moved: &Self::Value,
        _rel_table: &[f32], _sink: &Self::Value, _gate: &Self::Value, _steps: &Self::Value,
    ) -> Result<Option<Self::Value>> { Ok(None) }
    /// The attention of a few query rows in one launch (`FLASH_WINDOW_ATTN` in its exact form): the words of arms
    /// `HeadClip` (the query and the relative state), `RelBias`, `WindowScores`, `SinkSoftmax` and `WindowValues` of
    /// each row, bit for bit. `Ok(None)`: no such launch here; the caller runs the arms.
    /// The exact attention with separate key and value planes in chat order.
    fn window_planes(&self,_:&WindowOne<'_,Self::Value>,_:&[(Self::Value,Self::Value)])->Result<Option<Self::Value>> {Ok(None)}
    fn window_one(&self,_:&WindowOne<'_,Self::Value>)->Result<Option<Self::Value>> {Ok(None)}
    /// GQA-packed Split-KV tensor-core decode/verify attention with zero-copy provisional delta and multi-chat support.
    fn window_split(&self,_:&WindowSplit<'_,Self::Value>)->Result<Option<Self::Value>> {Ok(None)}
    /// `flash_window_attn` for the rows of a prompt with the clips inside the launch: `a.query` and `a.moved` are the
    /// products before their clips, and the words are those of arm `HeadClip` (the query and the relative state, with
    /// `a.query_scale`, `a.moved_scale`, `a.position` and the bounds), then `flash_window_attn` at `first_position`.
    /// `Ok(None)`: no such launch here; the caller clips, then runs `flash_window_attn`.
    fn window_rows(&self,_:&WindowOne<'_,Self::Value>,_first_position:u64)->Result<Option<Self::Value>> {Ok(None)}
    /// The key or the value row of a token in one launch: the ring fill, arm `ShortConv`, the clip or the norm, arm
    /// `PackBf16` and the plane fill, bit for bit; it gives the finished row unpacked. `Ok(None)`: no such launch
    /// here, nothing written; the caller runs the separate steps.
    fn conv_finish(&self,_:&ConvFinish<'_,Self::Value>,_ring:&mut Self::Value,_plane:&mut Self::Value)->Result<Option<Self::Value>> {Ok(None)}
    fn greedy_projection(&self,c:&CheckpointAdmission,name:&str,x:&Self::Value)->Result<u32> {
        let head=Mxfp8Projection::matrix(c,name,WeightType::F32)?;let values=self.mx(&head,x)?;
        greedy_logits(&self.read(&values)?,head.shape().rows())
    }
    /// DRAFT CONFIDENCE: `greedy_projection` and the probability of its pick, a score for the draft head's proposals
    /// (None where the backend makes none). The pick is `greedy_projection`'s.
    fn greedy_projection_mass(&self,c:&CheckpointAdmission,name:&str,x:&Self::Value)->Result<(u32,Option<f32>)> {
        let head=Mxfp8Projection::matrix(c,name,WeightType::F32)?;let values=self.mx(&head,x)?;
        greedy_logits_mass(&self.read(&values)?,head.shape().rows())
    }
    /// ONE PASS (proposer.rs `propose_chains`): `greedy_projection_mass` of `rows` rows of `x` (row-major) at once, each
    /// row's pick and probability as that call gives them for the row alone. None, before any work: no such form here;
    /// the caller picks each row alone. A wrapper hands this on.
    fn greedy_projection_mass_rows(&self,_:&CheckpointAdmission,_name:&str,_x:&Self::Value,_rows:usize)->Result<Option<Vec<(u32,Option<f32>)>>> {Ok(None)}
    fn projection_rows(&self,_:&CheckpointAdmission,_:&str,_:&[&Self::Value],_:bool)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn projection_batch(&self,_:&CheckpointAdmission,_:&[(&str,&Self::Value)],_:bool)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn projection_stage(&self,_:&CheckpointAdmission,_:&[crate::projection_stage::Job<'_,Self::Value>])->Result<Option<Vec<Self::Value>>> {Ok(None)}
    fn routed_experts(&self,_:&[engine_model::plan::layer::Step],_:&Self::Value,_:&Self::Value)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// The picked experts' down projection and its fold (`fold`, the ExpertFold shape of the whole output) in one stage:
    /// the fold's output, complete. `steps` is the one ExpertDown step; `shared` is the fold's last input when it has one.
    /// None: the caller runs the two steps apart.
    fn routed_fold(&self,_:&[engine_model::plan::layer::Step],_:&Self::Value,_:&Self::Value,_:&Self::Value,_:Option<&Self::Value>,_:Shape)->Result<Option<Self::Value>> {Ok(None)}
    /// True when `routed_fold` can run here (a device exchange is attached). Then a routed down step whose fold still
    /// waits on an input (the shared expert's row, after a refused earlier stage) runs with its fold at the fold step.
    fn folds_routed(&self)->bool {false}
    /// A kept projection of exactly these immutable rows. No device read.
    fn projection_origin(&self,_:&str,_:&Self::Value,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    /// `projection_origin` for rows `slots` of `rows` (each `width` words), in that order. A
    /// device answers all of them in one call and makes no view of a row it does not keep.
    fn projection_origins(&self,name:&str,rows:&Self::Value,width:usize,slots:&[usize],round:bool)->Result<Vec<Option<Self::Value>>> {
        slots.iter().map(|&slot|{let row=self.range(rows,slot.checked_mul(width).ok_or("kept row offset overflow")?,width)?;self.projection_origin(name,&row,round)}).collect()
    }
    /// The next call of `projection_origins` for `name` will ask for rows of a Gather of `rows`
    /// (`width` words a row), after a device read of the selected row ids. A device may make the
    /// answers for the rows of the last selection now, while it would only wait for that read.
    /// The answers and every memo count are those of the lookup after the read.
    fn prepare_origins(&self,_name:&str,_rows:&Self::Value,_width:usize,_round:bool)->Result<()> {Ok(())}
    /// Keep pure projections (input rows, output) for later tokens. The owner
    /// bounds the kept bytes; all pairs of one call come from one allocation.
    fn remember_projections(&self,_:&str,_:bool,_:&[(&Self::Value,&Self::Value)])->Result<()> {Ok(())}
    fn repeated_projection(&self,c:&CheckpointAdmission,name:&str,x:&Self::Value,round:bool)->Result<Option<Self::Value>> {self.projection(c,name,x,round)}
    fn attention(&self,_heads:u32,_key_dim:u32,_value_dim:u32,_slots:usize,_capacity:usize,_x:[&Self::Value;4])->Result<Option<Self::Value>> {Ok(None)}
    /// Required on wrappers: dropping this dispatch would silently restore the old arithmetic.
    fn has_latent_absorption(&self,name:&str)->bool;
    /// `slots` is the layer's row bound (never the live count): the two heads x rows vectors of the path are stored at
    /// heads x slots words, so a decode step takes vectors of one size at every context (the decode reserve reuses by exact size).
    fn latent_absorption(&self,name:&str,shape:Shape,query:&Self::Value,latent:&Self::Value,count:&Self::Value,slots:usize)->Result<Self::Value>;
    fn latent_absorption_rows(&self,name:&str,shape:spark_gpu_probe::prompt_ops::Shape,query:&Self::Value,latent:&Self::Value,selected:&Self::Value)->Result<Self::Value>;
    /// CHAT ROWS: the absorbed attention of a batch step's chats, one decode row each, as one row group of
    /// `latent_absorption_rows` (`shape.tokens` chats, the verify pass's kernels: each row keeps SUM-ORDER v1).
    /// `gathers[i]` is chat i's Gather as `gather_absorbed` takes it (shape, inputs, the count of the selection rule).
    /// It writes its rows at row i x `stride` of one latent value; `selected` names those rows and chat i's count.
    /// None: the caller runs the chats one by one. No default: a wrapper that dropped it would run one attention a chat.
    fn latent_absorption_chats(&self,name:&str,shape:spark_gpu_probe::prompt_ops::Shape,gathers:&[ChatGather<'_,Self::Value>],
        stride:usize,query:&Self::Value,selected:&Self::Value)->Result<Option<Self::Value>>;
    fn indirect_attention(&self,_shape:Shape,_parts:&[&Self::Value],_:&Self::Value)->Result<Option<Self::Value>> {Ok(None)}
    /// True when this compute runs the planar form of a KDA step or convolution (Op::StepPlanar, Op::ConvParts) for this shape: one
    /// launch whose result holds all heads' states then all outputs (or all convolution outputs then all histories), taken as two views.
    fn planar_kda(&self,_shape:Shape)->bool {false}
    fn embedding(&self,_:u32,_:usize)->Result<Option<Self::Value>> {Ok(None)}
    fn upload(&self,v:&[f32])->Result<Self::Value>;
    /// `upload` of words the host keeps: a later `read` of the whole value answers from the
    /// host's copy, with no wait for the device. The words on the device are the same.
    fn upload_known(&self,v:&[f32])->Result<Self::Value> {self.upload(v)}
    /// `upload` of words that no call writes after: a step word, a position factor. The compute may give one value
    /// for equal words, so a later call with the same words uploads nothing. The words on the device are the same.
    fn upload_constant(&self,v:&[f32])->Result<Self::Value> {self.upload(v)}
    fn read(&self,v:&Self::Value)->Result<Vec<f32>>;
    fn read_pair(&self,first:&Self::Value,second:&Self::Value)->Result<(Vec<f32>,Vec<f32>)> {Ok((self.read(first)?,self.read(second)?))}
    /// `row` is the whole selection output, known without its bytes (select_admission::known_selection).
    /// A backend may skip reading these values, if it still compares the count word before any result leaves.
    /// No default: a wrapper that drops this call would silently lose the saving.
    fn select_known(&self,ids:&Self::Value,count:&Self::Value,row:&[f32])->Result<()>;
    /// Multi-row counterpart of `select_known`: `rows` is the known `m * (slots + 1)` selection matrix
    /// (`select_admission::known_selection_rows`). Returns `true` when the backend deferred checking all `m`
    /// count words until the next real read so the caller can skip reading `value` mid-step.
    fn select_known_rows(&self,_value:&Self::Value,_slots:usize,_rows:&[f32])->Result<bool> {Ok(false)}
    fn range(&self,v:&Self::Value,start:usize,len:usize)->Result<Self::Value>;
    /// The words a value holds on this rank (a value of a recurrent layer holds this rank's own heads).
    fn words(&self,v:&Self::Value)->usize;
    /// A join whose storage has one size for every call of its kind (`reserve` words, the
    /// most the join can hold), so the storage of an earlier call can serve again.
    fn join_reserved(&self,parts:&[&Self::Value],_reserve:usize)->Result<Self::Value> {self.join(parts)}
    fn join(&self,parts:&[&Self::Value])->Result<Self::Value>;
    /// A scalar latent-cache append may keep bounded room beyond its immutable prefix.
    fn cache_append(&self,old:&Self::Value,new:&Self::Value,_rows:usize,_width:usize,_room:usize)->Result<Self::Value> {self.join(&[old,new])}

    fn zeros(&self,len:usize)->Result<Self::Value>;
    fn operation(&self,s:Shape,x:[Option<&Self::Value>;4],w:&[f32])->Result<Self::Value>;
    /// The Gather of an absorbed latent layer. The kernels read the count on the device and nothing reads the row
    /// identities, so the host waits for neither. `expected` is the count that the selection rule gives for this
    /// context. The compute compares it with the device's count word before a result leaves: a NaN score writes
    /// another count. Required on wrappers: a dropped dispatch would bring the blocking read back.
    fn gather_absorbed(&self,s:Shape,x:[Option<&Self::Value>;4],reserve:usize,expected:f32)->Result<Self::Value>;
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
    /// The residual add of a BF16 model, `round_bf16(x + y)` a word (the drafting layer's Combine). This form reads both,
    /// adds on the host and uploads the sum; a device may keep it on the device with the same words.
    fn bf16_sum(&self,x:&Self::Value,y:&Self::Value)->Result<Self::Value> {host_bf16_sum(self,x,y)}
    fn mean(&self,x:&Self::Value,streams:u32,hidden:u32)->Result<Self::Value>;
    /// One operation over token-major prompt rows.
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Self::Value>;5],w:&[f32])->Result<Self::Value>;
    fn rows_recurrent(&self,_:spark_gpu_probe::recurrent_rows::Shape,_:[&Self::Value;7],_:&Self::Value,
        _:[&[f32];3],_:bool)->Result<Option<RecurrentRows<Self::Value>>> {Ok(None)}
    /// These immutable operands have the same words and order as the unprepared row operation.
    fn rows_recurrent_prepared(&self,s:spark_gpu_probe::recurrent_rows::Shape,x:[&Self::Value;7],gate:&Self::Value,
        weights:[&Rc<StaticParameters>;3])->Result<Option<RecurrentRows<Self::Value>>> {
        self.rows_recurrent(s,x,gate,weights.map(|owner|owner.values()),true)
    }
    fn rows_embedding(&self,_:&[u32],_:usize)->Result<Option<Self::Value>> {Ok(None)}
    /// One stream per proposal, without starting a target row pass or trimming its workspace.
    fn draft_embeddings(&self,_:&[u32])->Result<Option<Self::Value>> {Ok(None)}
    /// Token-major `[m,rows]`, complete on every rank. None runs one row at a time.
    fn rows_project(&self,_:&CheckpointAdmission,_:&str,_:&Self::Value,_:usize,_:bool)->Result<Option<Self::Value>> {Ok(None)}
    /// A batch step of `chats` chats starts. `state_bytes`: the recurrent state one chat's step
    /// makes anew and then frees. A device keeps that much spare storage a chat between steps, so
    /// the new states of every chat take buffers the last step returned.
    fn batch_rows(&self,_chats:usize,_state_bytes:usize)->Result<()> {Ok(())}
    /// Several `rows_project` calls of one row group, as (matrix, rows, round), with one exchange
    /// for all of them. The same words as the single calls. None, before any work: make the calls singly.
    fn rows_stage(&self,_:&CheckpointAdmission,_:&[(&str,&Self::Value,bool)],_:usize)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// ROWS OVERLAP: as `rows_stage`, but the gather of block-FP8 outputs may wait for `rows_stage_close`; those outputs
    /// are None until then. None: not offered, and the caller runs `rows_stage`.
    fn rows_stage_open(&self,_:&CheckpointAdmission,_:&[(&str,&Self::Value,bool)],_:usize)->Result<Option<Vec<Option<Self::Value>>>> {Ok(None)}
    /// ROWS OVERLAP: the outputs `rows_stage_open` left None, by their job index.
    fn rows_stage_close(&self)->Result<Vec<(usize,Self::Value)>> {Err("no row stage is held".into())}
    /// One layer's selected experts in one pass. `first` and `second` list
    /// (matrix, token rows) over contiguous `source` rows. With `second`, the
    /// two local outputs meet in SwiGLU before the gather. `places` has three
    /// numbers for each output row. None runs each expert separately.
    fn rows_experts(&self,_:&[(&str,usize)],_:Option<(&[(&str,usize)],f32)>,_:&Self::Value,_:&[f32])->Result<Option<Self::Value>> {Ok(None)}
    /// GATHERED ROWS: `rows_experts` for a gate and up step whose rows were NOT copied into expert order. `source` holds
    /// one row a token; `order` has one number for each row the lists read: its row of `source` (the table of the
    /// `PermuteRows` the caller would run). The compute reads each row where it lies. The words of `rows_experts` on the
    /// copied rows. None, before any launch: the caller copies the rows and calls `rows_experts`.
    fn rows_experts_gathered(&self,_:&[(&str,usize)],_:Option<(&[(&str,usize)],f32)>,_:&Self::Value,_:&[f32],_:&[f32])->Result<Option<Self::Value>> {Ok(None)}
    /// LOOKAHEAD: `rows_experts` for the routed gate and up of a prompt group (`second` is given), stopped before the
    /// link step of its exchange. True: this rank's rows and their copy to the exchange are queued, and the exchange is
    /// open. The caller then queues work that does not read the result and makes no exchange, and MUST call
    /// `rows_experts_close`, also when that work failed. The work runs on this rank's GPU while the link carries the
    /// rows; before, the GPU ran nothing during a link step. False, before any work: the caller calls `rows_experts`.
    fn rows_experts_open(&self,_:&[(&str,usize)],_:(&[(&str,usize)],f32),_:&Self::Value,_:&[f32])->Result<bool> {Ok(false)}
    /// LOOKAHEAD: the link step and the result of the exchange that `rows_experts_open` left open: the value
    /// `rows_experts` gives. `failed`: the work queued ahead failed, and the exchange fails with it.
    fn rows_experts_close(&self,_failed:bool)->Result<Self::Value> {Err("no prompt exchange is open".into())}
    /// LOOKAHEAD over GATHERED ROWS: `rows_experts_open` for rows that were NOT copied into expert order (`source` and
    /// `order` as in `rows_experts_gathered`). True: the exchange is open and `rows_experts_close` MUST follow, as for
    /// `rows_experts_open`. False, before any launch: the caller goes on without it (`rows_experts_gathered`, then the
    /// copied rows).
    fn rows_experts_open_gathered(&self,_:&[(&str,usize)],_:(&[(&str,usize)],f32),_:&Self::Value,_:&[f32],_:&[f32])->Result<bool> {Ok(false)}
    /// ROUTE ROWS: one layer's routed experts for `m` rows with the picks left on the device. `steps` is the layer's
    /// two ExpertProject steps (gate, up: their local outputs meet in SwiGLU with `bound` before the gather) or its
    /// one ExpertDown step. `ids` holds `m * picks` expert IDs, row-major, as a device value; `input` holds one row a
    /// token (gate and up) or one a (token, pick) (down). Result: one row a (token, pick), in that order, complete on
    /// every rank: the words `rows_experts` gives, placed back to (token, pick). None, before any work: the caller
    /// reads the route and groups the rows by expert.
    fn rows_routed(&self,_:&[engine_model::plan::layer::Step],_:Option<f32>,_:&Self::Value,_:&Self::Value,_:usize)->Result<Option<Self::Value>> {Ok(None)}
    /// True when `rows_routed` can take the `m` rows of the row pass that runs now (the device compute: a batch step's
    /// rows after its first routed layer saw sparse expert sharing, with its device exchange attached). The route of
    /// such a row group then stays on the device.
    fn routes_rows(&self,_m:usize)->bool {false}
    /// ROWS FOLD: the routed down step of `m` rows and its fold as one stage (`rows_routed` for the one ExpertDown step
    /// `down`, then LOCAL FOLD a row: `routed_fold`'s work for each row, and ONE exchange of the folded rows). `ids` and
    /// `weights` hold `m * picks` words, row-major; `input` one row a (token, pick); `shared` the fold's last input, one
    /// row a token: the gathered rows, or this rank's own rows of them (`rows_project_local`); `fold` is the ExpertFold
    /// shape of one token's whole output. Result: the fold's output, one row a token, complete on every rank. None, before
    /// any work: the caller runs the two steps apart.
    #[allow(clippy::too_many_arguments)]
    fn rows_routed_fold(&self,_down:&engine_model::plan::layer::Step,_ids:&Self::Value,_input:&Self::Value,_weights:&Self::Value,_shared:Option<&Self::Value>,_fold:Shape,_m:usize)->Result<Option<Self::Value>> {Ok(None)}
    /// ROWS FOLD, the shared row: this rank's own rows of a row projection, token-major, with the words its exchange would
    /// carry, and NO exchange. For the one projection whose only reader is a fold that `rows_routed_fold` takes (the shared
    /// expert's down rows): that fold reads each rank's own rows only. None, before any work: the caller makes the whole
    /// rows with their exchange (`rows_project`).
    fn rows_project_local(&self,_:&CheckpointAdmission,_name:&str,_x:&Self::Value,_m:usize,_round:bool)->Result<Option<Self::Value>> {Ok(None)}
    /// LOCAL QUERY: `rows_project_local` for a projection whose readers take only the output rows `rows` (of one token's
    /// output) on this rank: this rank's own rows, token-major `[m, rows.len()]`, with the words its exchange would carry,
    /// and NO exchange. None, before any work: this rank's rows of the matrix are not exactly `rows`, or the compute keeps
    /// no rows on their rank for this pass. The caller then makes the whole rows with their exchange (`rows_project`).
    fn rows_project_owned(&self,_:&CheckpointAdmission,_name:&str,_x:&Self::Value,_m:usize,_round:bool,_rows:std::ops::Range<usize>)->Result<Option<Self::Value>> {Ok(None)}
    /// True when `rows_routed_fold` can take the `m` rows of the row pass that runs now (`routes_rows` with a device
    /// exchange attached). The routed down step of such a pass then waits for its fold.
    fn folds_rows(&self,_m:usize)->bool {false}
    /// GROUP FOLD: true when `rows_experts_fold` can take the `m` rows of a prompt group (the device compute: two ranks
    /// that split the rows). The grouped down step of such a group then waits for its fold.
    fn folds_groups(&self,_m:usize)->bool {false}
    /// GROUP FOLD: the grouped down step of a prompt group and its fold as one stage: `rows_experts` for the down step
    /// (`first`, `source`, `places` as it takes them), each rank's own rows folded on that rank, and ONE exchange of the
    /// folded rows (one row a token; the unfolded rows are one a (token, pick)). `ids` and `weights` hold `m * picks`
    /// words, row-major; `shared` is this rank's own rows of the fold's last input (`rows_project_local`); `fold` is the
    /// ExpertFold shape of one token's whole output. The fold is a chain a word, so the result holds the words of the fold
    /// after the gather: one row a token, complete on every rank. None, before any work: the caller runs the two steps apart.
    #[allow(clippy::too_many_arguments)]
    fn rows_experts_fold(&self,_first:&[(&str,usize)],_source:&Self::Value,_places:&[f32],_ids:&Self::Value,_weights:&Self::Value,_shared:Option<&Self::Value>,_fold:Shape,_m:usize)->Result<Option<Self::Value>> {Ok(None)}
    /// WIDE COHORT: `rows_experts_gathered` for the gate and up step of TWO OR MORE row sets (the runs of a wide prompt
    /// group), with ONE launch of the routed expert products a list for all the sets: the sets read each expert's
    /// weights from memory once. Each set is (gate list, up list, SwiGLU bound, source, order, places), as
    /// `rows_experts_gathered` takes them. One result a set, each the words its own `rows_experts_gathered` gives.
    /// None, before any launch: the caller calls `rows_experts_gathered` for each set.
    #[allow(clippy::type_complexity)] // the arguments of `rows_experts_gathered`, a tuple a set.
    fn rows_experts_gathered_wide(&self,_sets:&[(&[(&str,usize)],&[(&str,usize)],f32,&Self::Value,&[f32],&[f32])])->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// PACKED MIDDLE: `rows_experts_gathered_wide` when the ONLY reader of each set's result is `rows_experts_fold_wide`
    /// with the down list `down[set]` (the caller passes each result to that call as its source and to nothing else).
    /// A compute may then give each result in a private form of that fold: the device sends each rank's packed operand
    /// bytes of the down products (4-bit codes and scale codes) over the link in place of FP32 words, and the fold's
    /// product launch reads them. `reader` gives the actual fold shape and its shared projection's name, so a device
    /// checks the fold admission and local shared-row ownership before packing.
    /// The fold's words are the words of the plain form. Default: the plain form.
    #[allow(clippy::type_complexity)] // the arguments of `rows_experts_gathered`, a tuple a set.
    fn rows_experts_gathered_wide_for(&self,sets:&[(&[(&str,usize)],&[(&str,usize)],f32,&Self::Value,&[f32],&[f32])],_down:&[&[(&str,usize)]],_reader:(Shape,Option<&str>))->Result<Option<Vec<Self::Value>>> {self.rows_experts_gathered_wide(sets)}
    /// WIDE COHORT: `rows_experts_fold` for the down step of TWO OR MORE row sets, with ONE launch of the routed expert
    /// products for all the sets. Each set is (down list, source, places, ids, weights, shared, fold, m), as
    /// `rows_experts_fold` takes them. One result a set, each the words its own `rows_experts_fold` gives.
    /// None, before any launch: the caller calls `rows_experts_fold` for each set.
    #[allow(clippy::type_complexity)] // the arguments of `rows_experts_fold`, a tuple a set.
    fn rows_experts_fold_wide(&self,_sets:&[(&[(&str,usize)],&Self::Value,&[f32],&Self::Value,&Self::Value,Option<&Self::Value>,Shape,usize)])->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// Build stable expert tables on the device. Sets hold (rows, source, IDs).
    /// The private middle belongs only to the matching down/fold call. None
    /// refuses the complete stage before any route or product is submitted.
    #[allow(clippy::type_complexity)]
    fn rows_experts_device_wide(&self,_steps:&[engine_model::plan::layer::Step],_bound:f32,_sets:&[(usize,&Self::Value,&Self::Value)],_reader:(Shape,Option<&str>))->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// Consume the private middle with the same immutable route owner. Sets
    /// hold (rows, middle, IDs, route weights, local shared rows).
    #[allow(clippy::type_complexity)]
    fn rows_experts_fold_device_wide(&self,_down:&engine_model::plan::layer::Step,_sets:&[(usize,&Self::Value,&Self::Value,&Self::Value,Option<&Self::Value>)],_fold:Shape)->Result<Option<Vec<Self::Value>>> {Ok(None)}
    /// After a batch step's first routed layer read `picks` total picks across `distinct` experts: sparse sharing
    /// (`picks < 2 * distinct`, below the tensor-core cohort admission line of `expert_group_launch`) switches the
    /// step's remaining routed layers to `rows_routed`; dense sharing keeps the grouped tensor-core form (`nv_rows_tc`).
    fn note_batch_experts(&self,_picks:usize,_distinct:usize) {}
    /// Bracket a multi-row verify pass (`2..=VERIFY_ROWS` rows): its routed layers stay on the device with zero mid-step waits.
    fn rows_verify(&self,_on:bool) {}
    fn warm_routed_rows(&self,_:&[engine_model::plan::layer::Step],_picks:usize,_max_rows:usize)->Result<()> {Ok(())}
    /// A large group has ended: its spare working memory can go.
    fn rows_release(&self)->Result<()> {Ok(())}
    /// After a prompt group: keep its spares within the pool's normal room (default: release).
    fn rows_keep(&self)->Result<()> {self.rows_release()}
    fn prompt_spare_bytes(&self) -> Option<usize> { None }
    fn return_prompt_spares(&self, _target: usize) -> Result<(usize, usize)> { Ok((0, 0)) }
    fn rows_cohort_spares_begin(&self)->Result<()> {Ok(())}
    fn rows_cohort_chat(&self,_chat:Option<usize>) {}
    fn rows_cohort_spares_end(&self)->Result<()> {Ok(())}
    /// Return a temporary prompt workspace and report actual spare ownership.
    fn rows_return_lease(&self)->Result<()> {self.rows_keep()}
    /// WIDE PHASE: a prompt group above the whole-charged lease sizes ended a mixer run or a layer. Submit what is
    /// queued, wait for it, and free every buffer dropped before: the group's charge counts none of them
    /// (`wide_phase_workspace`, rows/wide_charge.rs). Carries, states and pool spares stay. Default: the group's end, which drains
    /// too (a compute that forgets this call trims its spares; it never skips the drain). No row arithmetic changes.
    fn rows_phase_end(&self)->Result<()> {self.rows_keep()}
    /// Pause (true) or resume (false) launch recording around a prompt rows pass; returns the
    /// previous paused value. A compute that records nothing ignores it.
    fn rows_unrecorded(&self,_pause:bool)->Result<bool> {Ok(false)}
    /// The last group of a prompt is done and decode steps follow. A device trims its vector pool here, off the
    /// first decode step. No default: a wrapper that drops this call would lose the saving without a word.
    fn prompt_end(&self)->Result<()>;
    /// True when the last group of a prompt leaves the rows it expanded in the projection memo (`Decoder::set_memo_seed`).
    /// No default, as above.
    fn prompt_seeds(&self)->bool;
    /// The device bytes memory admission charged for the sessions now live, told before a prompt group: the spare
    /// buffers a large group may keep (`group_pool_cap`) stay under it. A service that has such a charge calls this
    /// for every group; a tool without admission does not, and its groups have the cap alone. Nothing off the device.
    fn rows_charge(&self,_bytes:usize) {}
    /// GROUP SPARES. `follows`: the prompt of the group that runs next goes on after it (its service reads the prompt in
    /// consecutive groups and says so before each one). The end of such a group then leaves the vector pool's spares and
    /// the group's cap as they are (`rows_keep` frees nothing): the next group takes the same buffer sizes back, where
    /// each group's end freed them and each group's start bought them again (measured on a pair, an 8,192-token prompt
    /// in 8 groups: 790 frees and 999 allocations, 146 ms and 213 ms with the device idle). Free storage only, inside
    /// the group's charge (`rows_charge`): the pool's own rules release it before an allocation passes that charge. A
    /// decode token, a small group and a session's first group trim the pool at their start, as before, and
    /// `rows_settle` trims it when a session leaves. Nothing off the device.
    fn rows_follow(&self,_follows:bool) {}
    /// GROUP SPARES: a session leaves. Spares that a group's end left for a next group (`rows_follow`) are trimmed now, as
    /// that group's end would have trimmed them: the pool at rest holds what it held.
    fn rows_settle(&self)->Result<()> {Ok(())}
    /// PROJ-DELIVERY: called before each plain decode step. True when a latent layer of that step reads its selection
    /// back (the host cannot know it at this context, so the step waits for the device). Speed only: a compute keeps its
    /// projection side lane off for such a step, because a fork there waits behind the selection reads.
    fn decode_selection_reads(&self,_reads:bool) {}
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
    /// Write `part` at `offset` in a value only this caller holds, in stream order and with no wait for the
    /// device: a launch queued before it reads the old words, a launch queued after it reads the new ones. For a
    /// state that a decoder writes in place. The body here is `rows_fill`.
    fn state_fill(&self,target:&mut Self::Value,offset:usize,part:&Self::Value)->Result<()> {let words=self.words(part);self.rows_fill(target,offset,part,words)}
    /// Writable state plane of `full` words with `live` copied at offset 0 (unwritten tail is written before any read).
    fn state_restore(&self,live:&Self::Value,full:usize)->Result<Self::Value> {let mut plane=self.zeros(full)?;self.state_fill(&mut plane,0,live)?;Ok(plane)}
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
/// A second handle of every plane of `state`, so that the state can be KEPT while a chat goes on from it.
/// A step and a prompt group read their old state and return a new one; they never write the old planes.
/// On the device no word is copied: `Compute::range` of a whole value is a new handle of the same view,
/// a view stays live while any handle of it does, and an append writes only past the last live view or
/// copies (held.rs). The host twin copies its vectors, which is the same words.
#[cold] #[inline(never)]
fn share_layers<C:Compute>(compute:&C,state:&State<C::Value>)->Result<Vec<BTreeMap<String,C::Value>>> {
    state.layers.iter().map(|planes|planes.iter().map(|(name,value)|Ok((name.clone(),compute.range(value,0,compute.words(value))?))).collect::<Result<BTreeMap<_,_>>>()).collect()
}
#[cold] #[inline(never)]
pub fn share_state<C:Compute>(compute:&C,state:&State<C::Value>)->Result<State<C::Value>> {
    Ok(State {position:state.position,layers:share_layers(compute,state)?,expanded:state.expanded.clone()})
}
/// A kept prompt while no chat runs from it. It holds a second handle of every plane (`share_state`) but NOT the
/// expanded rows: of those it keeps only how many rows each layer held, and in which layout. The expanded rows are
/// pure projections of `latent_cache` rows, which a parked prompt keeps, so `Decoder::resume_state` builds the
/// same words again. A parked prompt therefore holds its state planes and nothing else.
pub struct Parked<V> {state:State<V>,held:BTreeMap<usize,(usize,crate::sparse_heads::Layout)>,owners:Option<owners::Owners>}
impl<V> std::ops::Deref for Parked<V> {type Target=State<V>;fn deref(&self)->&State<V> {&self.state}}
impl<V> Parked<V> {
    /// Page-rounded owners of these planes, counted once within this holder.
    pub fn retained_bytes(&self)->Option<usize> {self.owners.as_ref().map(owners::Owners::bytes)}
    pub fn owners(&self)->Option<&owners::Owners> {self.owners.as_ref()}
    /// The most expanded rows any layer held when the prompt was parked: what a resume builds again for each such layer.
    #[cold] #[inline(never)]
    pub fn held_rows(&self)->usize {self.held.values().map(|held|held.0).max().unwrap_or(0)}
}
/// Park `state`: share its planes and drop its expanded rows. No word is copied and nothing is launched.
#[cold] #[inline(never)]
pub fn park_state<C:Compute>(compute:&C,state:&State<C::Value>)->Result<Parked<C::Value>> {
    let held=state.expanded.iter().map(|(&layer,rows)|(layer,(rows.1,rows.2))).collect();
    let state=State {position:state.position,layers:share_layers(compute,state)?,expanded:BTreeMap::new()};
    let owners=state_owners(compute,&state);
    Ok(Parked {state,held,owners})
}
/// No device read, launch, or synchronization. Expanded projections are excluded.
#[cold] #[inline(never)]
pub fn state_owner_bytes<C:Compute>(compute:&C,state:&State<C::Value>)->Option<usize> {
    state_owners(compute,state).map(|owners|owners.bytes())
}
/// Cold cache quote or park only. The saved parked list serves all later reads.
#[cold] #[inline(never)]
pub fn state_owners<C:Compute>(compute:&C,state:&State<C::Value>)->Option<owners::Owners> {
    let planes=state.layers.iter().map(BTreeMap::len).sum::<usize>();
    let mut owners=Vec::with_capacity(planes.saturating_mul(2));
    for value in state.layers.iter().flat_map(|planes|planes.values()) {if !compute.state_allocations(value,&mut owners) {return None;}}
    Some(owners::Owners::new(owners))
}
#[cfg(test)]
mod kept_owner_tests {
    use super::*;
    #[test]
    fn each_owner_is_rounded_and_counted_once_per_holder() {
        let owners=[(1,4097),(1,4097),(2,1),(3,0)];
        assert_eq!(owners::Owners::new(owners.to_vec()).bytes(),12288);
        let mut ledger=owners::Ledger::new();let owners=owners::Owners::new(owners.to_vec());
        ledger.retain(&owners);ledger.retain(&owners);assert_eq!(ledger.bytes(),12288);
        assert_eq!(owners::Owners::new(vec![(1,usize::MAX)]).bytes(),usize::MAX);
    }
    #[test]
    fn parking_charges_plane_capacity_and_excludes_expanded_rows() {
        let mut value=Vec::with_capacity(2048);value.push(0.0);
        let layout=crate::sparse_heads::Layout::new(1,1,1,Topology {rank:0,parts:1}).unwrap();
        let state=State {position:1,layers:vec![BTreeMap::from([("plane".into(),value)])],expanded:BTreeMap::from([(0,Expanded(Rc::new(vec![0.0;4096]),1,layout))])};
        assert_eq!(state_owner_bytes(&Host,&state),Some(8192));
        // The host copies only the used words; the device keeps owner metadata.
        let parked=park_state(&Host,&state).unwrap();assert_eq!(parked.retained_bytes(),Some(4096));
    }
    #[test]
    fn parked_decoder_does_not_clone_program_schedules() {
        let checkpoint=host::shadow_fixture::Checkpoint::new_with_predictor(80);
        let mut live=Decoder::new(&checkpoint.admission,Host,80).unwrap();live.enable_predictor().unwrap();let prepared=live.prepared();
        assert!(!prepared.schedules.is_empty() && prepared.draft.is_some());
        let idle=Decoder::from_prepared_idle(&prepared,&Host,80).unwrap();
        assert_eq!(idle.schedules.borrow().capacity(),0);assert!(idle.draft_schedule.borrow().is_none());
        assert!(Rc::ptr_eq(&idle.parameters,&prepared.parameters));assert!(Rc::ptr_eq(&idle.blocks,&prepared.blocks));
        println!("KEPT HOST MEASURED schedule_clone_payload_bytes=0 idle_decoder_inline_bytes={}",std::mem::size_of_val(&idle));
    }
}
/// Host copies of immutable parameters, shared by all sessions of one server.
#[derive(Clone)]
/// Immutable parameters of one checkpoint. They stay bound to it: a decoder built
/// from them reads that checkpoint's weights and can read no other.
pub struct Prepared<'f> {topology:Topology,checkpoint:&'f CheckpointAdmission,parameters:Rc<BTreeMap<String,Vec<f32>>>,static_parameters:Rc<StaticParameterPlan>,initial_state:Rc<initial_state::Layouts>,blocks:Rc<BTreeMap<(usize,usize),crate::layer_operands::ResidualBlockOperands>>,schedules:Vec<LayerSchedule>,draft:Option<(Rc<proposer::Draft>,LayerSchedule)>,verify_routes:usize}
impl Prepared<'_> {
    /// Count the actual prepared owners that can retain fused scratch. Charge the full allocation even if an
    /// owner is already resident or the compute lane does not use fused kernels. This is a conservative bound.
    pub fn fused_boundary_scratch(&self)->Result<(usize,usize)> {
        use spark_gpu_probe::mhc_params::{Shape,fused_admitted,FUSED_SCRATCH_BYTES};
        let owners=self.blocks.values().filter(|p|fused_admitted(Shape {rows:1,..p.mhc().shape()})).count();
        Ok((owners,owners.checked_mul(FUSED_SCRATCH_BYTES).ok_or("fused boundary scratch charge overflow")?))
    }
}
pub struct Decoder<'f,C:Compute> {checkpoint:&'f CheckpointAdmission,compute:C,capacity:u64,parameters:Rc<BTreeMap<String,Vec<f32>>>,static_parameters:Rc<StaticParameterPlan>,initial_state:Rc<initial_state::Layouts>,
    blocks:Rc<BTreeMap<(usize,usize),crate::layer_operands::ResidualBlockOperands>>,schedules:RefCell<Vec<LayerSchedule>>,expanded_limit:std::cell::Cell<usize>,forced_union:std::cell::Cell<Option<usize>>,memo_seed:std::cell::Cell<bool>,
    /// For each layer, one plan a batch step's shared steps read (see batch.rs). Made at first use.
    batch_plans:RefCell<BTreeMap<usize,Rc<LayerPlan>>>,
    draft:Option<Rc<proposer::Draft>>,draft_schedule:RefCell<Option<LayerSchedule>>,verify_routes:std::cell::Cell<usize>,
    /// ROWS FOLD: for each layer (and the predictor layer behind them), the fold steps of its program, found at first use
    /// (`Decoder::fold_links` in rows.rs). None: not looked for yet.
    fold_links:Vec<std::cell::Cell<Option<Option<rows::FoldLinks>>>>}
impl<'f,C:Compute> Decoder<'f,C> {
    fn kda_layout(&self,layer:usize)->Result<crate::sparse_heads::Layout> {crate::kda_heads::layout(self.mixer(layer)?,self.compute.sparse_topology())}
    // The divisor is a variable. `%` keeps the panic on a zero head count; `is_multiple_of` would return an error there.
    #[allow(clippy::manual_is_multiple_of)]
    fn state_words(&self,layer:usize,words:usize)->Result<usize> {
        if matches!(self.mixer(layer)?,TokenMixer::Kda(_)) {let layout=self.kda_layout(layer)?;need(words%layout.total_heads()==0,"recurrent state head extent differs")?;Ok(words/layout.total_heads()*layout.local_heads())} else {Ok(words)}
    }
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
        // LOCAL QUERY: the rows of a prompt group that stayed on their rank (`rows_query_local`) are this value already.
        // The two extents differ (a rank owns some heads, not all), so the extent names the form.
        if self.compute.words(input)==tokens*rows.len() {return self.compute.range(input,0,tokens*rows.len());}
        if tokens==1 {return self.compute.range(input,rows.start,rows.len());}
        let parts=(0..tokens).map(|token|self.compute.range(input,token*width+rows.start,rows.len())).collect::<Result<Vec<_>>>()?;
        self.compute.join(&parts.iter().collect::<Vec<_>>())
    }
    #[inline]
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
        for (layer,boundary) in LayerBoundary::each(d,checkpoint.headers(),&storage,1,topology,usize::MAX).enumerate() {let boundary=boundary?;
            residual_plan::check(boundary.program())?;
            schedules.push(LayerSchedule::new(d,layer,topology).map_err(|e|e.to_string())?);
            for (index,step) in boundary.program().steps.iter().enumerate() {
                if StaticParameterPlan::admits(step) {parameter_jobs.push((layer,step.id.clone(),step.weights.iter().map(|w|w.name.clone()).collect()));}
                if step.kind==Kind::MhcPrepare {blocks.insert((layer,index),boundary.prepare_residual_block_checkpoint(index,checkpoint)?);}
                for w in &step.weights {if w.storage==WeightStorage::Plain && !parameters.contains_key(&w.name) {parameters.insert(w.name.clone(),plain(checkpoint,&w.name)?);}}
            }
            if matches!(d.layers[layer].mixer,TokenMixer::Kda(_)) {crate::kda_heads::parameters(boundary.program(),crate::kda_heads::layout(d.layers[layer].mixer,compute.sparse_topology())?,&mut parameters)?;}
        }
        let name="model.language_model.norm.weight";parameters.insert(name.into(),plain(checkpoint,name)?);
        let parameters=Rc::new(parameters);let static_parameters=Rc::new(StaticParameterPlan::new(parameters.clone(),parameter_jobs)?);
        crate::startup::step("weights");
        compute.prepare(checkpoint)?;Ok(Self {checkpoint,compute,capacity,parameters,static_parameters,initial_state:Rc::new(initial_state::Layouts::default()),blocks:Rc::new(blocks),schedules:RefCell::new(schedules),expanded_limit:std::cell::Cell::new(rows::RESERVED_ROWS),forced_union:std::cell::Cell::new(None),memo_seed:std::cell::Cell::new(false),batch_plans:RefCell::new(BTreeMap::new()),draft:None,draft_schedule:RefCell::new(None),verify_routes:std::cell::Cell::new(0),fold_links:(0..=checkpoint.description().layers.len()).map(|_|std::cell::Cell::new(None)).collect()})
    }
    /// Toggle already prepared predictor metadata; no weights or device scratch are allocated.
    /// The caller first drops this session's predictor state and growing held rows when disabling.
    pub(crate) fn set_prepared_predictor(&mut self,prepared:&Prepared<'f>,enabled:bool) {
        self.draft=prepared.draft.as_ref().filter(|_|enabled).map(|draft|draft.0.clone());
        *self.draft_schedule.borrow_mut()=prepared.draft.as_ref().filter(|_|enabled).map(|draft|draft.1.clone());
    }
    pub fn prepared(&self)->Prepared<'f> {Prepared {topology:self.compute.sparse_topology(),checkpoint:self.checkpoint,parameters:self.parameters.clone(),static_parameters:self.static_parameters.clone(),initial_state:self.initial_state.clone(),blocks:self.blocks.clone(),schedules:self.schedules.borrow().clone(),draft:self.draft.clone().zip(self.draft_schedule.borrow().clone()),verify_routes:self.verify_routes.get()}}
    /// A new session without rereading immutable parameters.
    /// `live`: the sessions that already decode with this owner (0: the new session is alone).
    pub fn from_prepared(prepared:&Prepared<'f>,compute:C,capacity:u64,live:usize)->Result<Self> {
        let checkpoint=prepared.checkpoint;need(prepared.topology==compute.sparse_topology(),"prepared recurrent ownership differs")?;
        need(capacity>0 && capacity<=checkpoint.description().max_positions,"bounded decoder capacity exceeds the recipe")?;
        compute.prepare_beside(checkpoint,live)?;
        Ok(Self::bound(prepared,compute,capacity))
    }
    /// The owner of a session that only HOLDS a kept state (`row_split::Session::fork`). It runs no step and no
    /// prompt group, so it prepares nothing on the device: the sessions that decode keep their memo as it is.
    /// Crate only: the caller passes the `prepared` and the compute of a session that `from_prepared` built,
    /// so the ownership check there has already passed for this pair.
    pub(crate) fn from_prepared_idle(prepared:&Prepared<'f>,compute:C,capacity:u64)->Result<Self> {
        need(capacity>0 && capacity<=prepared.checkpoint.description().max_positions,"bounded decoder capacity exceeds the recipe")?;
        // A parked holder never executes a schedule. A resumed live session
        // gets its schedules from Prepared in from_prepared, as before.
        Ok(Self {checkpoint:prepared.checkpoint,compute,capacity,parameters:prepared.parameters.clone(),static_parameters:prepared.static_parameters.clone(),initial_state:prepared.initial_state.clone(),blocks:prepared.blocks.clone(),schedules:RefCell::new(Vec::new()),expanded_limit:std::cell::Cell::new(rows::RESERVED_ROWS),forced_union:std::cell::Cell::new(None),memo_seed:std::cell::Cell::new(false),batch_plans:RefCell::new(BTreeMap::new()),draft:prepared.draft.as_ref().map(|d|d.0.clone()),draft_schedule:RefCell::new(None),verify_routes:std::cell::Cell::new(prepared.verify_routes),
            // No kept links either: a holder runs no row pass, and `fold_links` in rows.rs finds them again when a slot is absent.
            fold_links:Vec::new()})
    }
    fn bound(prepared:&Prepared<'f>,compute:C,capacity:u64)->Self {
        Self {checkpoint:prepared.checkpoint,compute,capacity,parameters:prepared.parameters.clone(),static_parameters:prepared.static_parameters.clone(),initial_state:prepared.initial_state.clone(),blocks:prepared.blocks.clone(),schedules:RefCell::new(prepared.schedules.clone()),expanded_limit:std::cell::Cell::new(rows::RESERVED_ROWS),forced_union:std::cell::Cell::new(None),memo_seed:std::cell::Cell::new(false),batch_plans:RefCell::new(BTreeMap::new()),draft:prepared.draft.as_ref().map(|d|d.0.clone()),draft_schedule:RefCell::new(prepared.draft.as_ref().map(|d|d.1.clone())),verify_routes:std::cell::Cell::new(prepared.verify_routes),fold_links:(0..=prepared.checkpoint.description().layers.len()).map(|_|std::cell::Cell::new(None)).collect()}
    }
    pub fn zero_state(&self)->Result<State<C::Value>> {self.zero_state_at(0)}
    /// Zero state of the shape a chat has at this position. Position zero is a new chat; any
    /// other position is for memory tests, because zeros are not what a prompt leaves there.
    pub fn zero_state_at(&self,position:u64)->Result<State<C::Value>> {
        // Only immutable full-head names and extents are kept with this checkpoint. Every call still applies
        // this compute owner's head layout and allocates fresh zeros. Diagnostic positions keep their own plans.
        let layouts=self.initial_state.get(position,|| {
            (0..self.checkpoint.description().layers.len()).map(|layer| {
                let p=LayerPlan::new(self.checkpoint.description(),layer,position+1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
                p.inputs.iter().filter(|b|b.name!="residual" && b.name!="local_slots")
                    .map(|b|Ok((b.name.clone(),b.elements().map_err(|e|e.to_string())?))).collect::<Result<Vec<_>>>()
            }).collect::<Result<Vec<_>>>()
        })?;
        let mut layers=Vec::with_capacity(layouts.len());for (layer,layout) in layouts.iter().enumerate() {
            let mut inputs=BTreeMap::new();for (name,words) in layout {inputs.insert(name.clone(),self.compute.zeros(self.state_words(layer,*words)?)?);}layers.push(inputs);
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
        self.step_greedy_hidden(token,state,observe).map(|(best,next,_)|(best,next))
    }
    /// The same greedy step, with the row's last hidden state before and after the final norm
    /// (two views, no launch). A verify pass of one row runs this: it is the plain decode step.
    pub fn step_greedy_hidden(&self,token:u32,state:&State<C::Value>,observe:impl FnMut(usize,&C::Value)->Result<()>)->Result<(u32,State<C::Value>,[C::Value;2])> {
        let (next,hidden)=self.step_body(token,state,observe,true)?;
        let best=self.compute.greedy_projection(self.checkpoint,"lm_head.weight",&hidden[1])?;
        need(u64::from(best)<self.checkpoint.description().vocab,"bounded greedy ID exceeds vocabulary")?;Ok((best,next,hidden))
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
        let context=state.position+1;
        self.compute.decode_selection_reads(d.layers.iter().any(|spec|matches!(spec.mixer,TokenMixer::LatentAttention(a) if selection_read(&a.indexer,context))));
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
    #[inline]
    fn pack_expanded(&self,value:&C::Value,words:usize)->Result<C::Value> {
        need(words>0 && words.is_multiple_of(2),"packed expanded rows need an even word count")?;
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
                // TENSOR-CORE REBASE (block FP8): the contract word of the format, computed on the host from the checkpoint's
                // bytes (`matrix_role::fp8_host_product`): the word the resident kernels give for this matrix on every path.
                // It is rounded to BF16, as on the device. (The one chain of scaled weights of `causal_ops::Op::Project`
                // is not used.)
                let word=crate::matrix_role::fp8_host_product(rows,cols,&u.weight,&u.scales,block[0] as usize,block[1] as usize,&self.compute.read(input)?)?;
                self.compute.upload(&word)?
            },
            WeightStorage::Nvfp4 {..}=>{
                // TENSOR-CORE REBASE: the contract word of the matrix role, computed on the host from the packed bytes
                // (`matrix_role::nv_host_product`): the word the resident kernels give for this matrix. The input is read
                // to the host for its quantizer, as before. (The rc.8 chain of `causal_ops::Op::Nvfp4` is not used.)
                let word=crate::matrix_role::nv_host_product(rows,cols,&u.weight,&u.scales,u.global_scale.ok_or("NVFP4 weight global scale is absent")?,
                    u.input_scale.ok_or("NVFP4 calibrated input scale is absent")?,crate::matrix_role::nv_role(name),&self.compute.read(input)?)?;
                self.compute.upload(&word)?
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
        self.layer_steps(layer,first_context,residuals,state,expanded,false,shadow,None)
    }
    /// `state_only`: the steps run until the layer's next state is complete (the caches a later
    /// position reads). What follows them, the attention itself and the experts, is left out,
    /// and no output is returned. For a row whose output nobody reads.
    /// `shadow`: the plain decode step (one row); see `step_purpose`.
    /// `span`: only these steps of one row's program, on values the caller holds and with no
    /// residual (the steps of a batch step that read one chat's history; see batch.rs). The
    /// results join those values. No output and no next state come back.
    #[allow(clippy::too_many_arguments)]
    fn layer_steps(&self,layer:usize,first_context:u64,residuals:&[&C::Value],state:&BTreeMap<String,C::Value>,expanded:Option<&Expanded<C::Value>>,state_only:bool,shadow:bool,mut span:Option<(std::ops::Range<usize>,&mut BTreeMap<String,C::Value>)>)->Result<(Vec<C::Value>,BTreeMap<String,C::Value>)> {
        if let Some(owner)=expanded {need(owner.2==self.sparse_layout(layer)?,"expanded owner layout differs")?;}
        need(if span.is_some() {residuals.is_empty()} else {!residuals.is_empty() && residuals.len()<=PROMPT_ROWS},"bounded prompt row count differs")?;
        let count=residuals.len().max(1);let one_row=count==1;let c=self.checkpoint;let d=c.description();let b=&self.compute;
        need(first_context>0 && first_context.checked_add(count as u64-1).is_some_and(|n|n<=self.capacity && n<=u32::MAX as u64),"bounded layer context exceeds capacity")?;
        let planning=std::time::Instant::now();
        // The appended predictor layer follows the decoder layers. It has one residual stream:
        // its prepare is the norm alone and its combine is a BF16 add (see proposer.rs).
        let drafting=layer==d.layers.len();let mixer=self.mixer(layer)?;let streams=if drafting {1} else {d.residual.streams as usize};
        let mut main=self.schedules.borrow_mut();let mut extra=self.draft_schedule.borrow_mut();
        // Catch-up reads only the cache producers. In particular, the projection
        // scheduler cannot bring query projections into this slice.
        let mut cache=self.draft.as_ref().filter(|_|drafting && state_only)
            .and_then(|draft|draft.cache_schedule.as_ref()).map(RefCell::borrow_mut);
        let schedule=if let Some(cache)=cache.as_deref_mut() {cache}
            else if drafting {extra.as_mut().ok_or("predictor layer is not enabled")?}
            else {main.get_mut(layer).ok_or("bounded layer is outside recipe")?};
        let owned_plans:Vec<LayerPlan>;
        let plans:Vec<&LayerPlan>=if count==1 {vec![schedule.at(first_context).map_err(|e|e.to_string())?]} else {
            owned_plans=(0..count).map(|row|schedule.at(first_context+row as u64).cloned().map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
            owned_plans.iter().collect()
        };
        b.plan_cost("planning",planning.elapsed().as_nanos());
        let names=match mixer {
            TokenMixer::Kda(_)=>vec![("kda.history","next_history"),("kda.state","next_state")],
            // The complete pools ride with the state. The state plan already holds their plane.
            TokenMixer::LatentAttention(_)=>vec![("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates"),("pool_keys","pool_keys")],
        };
        let mut all_values=(0..count).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        for (values,residual) in all_values.iter_mut().zip(residuals) {values.insert("residual".to_owned(),b.range(residual,0,d.hidden as usize*streams)?);}
        let mut all_passes=(0..count).map(|_|BTreeMap::new()).collect::<Vec<_>>();let mut skip=0;let mut projected=std::collections::BTreeSet::new();
        let mut steps=0..plans[0].steps.len();
        if let Some((part,held))=span.as_mut() {need(part.end<=steps.end,"bounded step span exceeds the program")?;steps=part.clone();all_values[0]=std::mem::take(&mut **held);
            // A projection of the span whose output the caller already holds is done: it is a pure function of its rows, and
            // the batch step's shared stage made it for every chat (the shared expert's gate and up; see batch.rs ROUTED CHATS).
            projected.extend(steps.clone().filter(|&at| {let step=&plans[0].steps[at];step.kind==Kind::Project && step.outputs.iter().all(|output|all_values[0].contains_key(&output.name))}));
        }
        // A routed down step deferred to its fold runs at that fold step, which must lie inside this run of steps.
        let mut deferred=BTreeMap::new();let steps_end=steps.end;
        for index in steps {
            if state_only && names.iter().all(|(_,output)|all_values[count-1].contains_key(*output)) {break;}
            if skip>0 {skip-=1;continue;}
            if projected.remove(&index) {continue;}
            let first=&plans[0].steps[index];
            need(plans.iter().all(|p|p.steps.len()==plans[0].steps.len() && p.steps[index].kind==first.kind && p.steps[index].id==first.id),"prompt row programs differ")?;
            // A routed down step deferred to its fold (see below) runs here, with the fold's every input ready.
            if let Some(down)=deferred.remove(&index) {
                if let Some(folded)=self.expert_fold(down,&plans[0],&mut all_values[0])? {need(folded==index,"deferred routed fold moved")?;continue;}
                if !self.expert_batch(down,1,&plans[0],&mut all_values[0])? {self.expert_rows(down,&plans,&mut all_values)?;}
            }
            // A routed down step is never staged: a stage exchanges the unfolded rows, LOCAL FOLD the folded ones.
            if count==1 && matches!(first.kind,Kind::Project|Kind::ExpertProject) {
                let done=self.projection_stage(index,steps_end,&plans[0],&mut all_values[0])?;
                if !done.is_empty() {projected.extend(done);continue;}
            }
            if first.kind==Kind::Project {
                for row in 0..count {self.import_row_state(layer,row,index,&plans,&names,state,&mut all_values)?;}
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
                    if span==1 && let Some(folded)=self.expert_fold(index,&plans[0],&mut all_values[0])? {projected.insert(folded);continue;}
                    if span==1 && self.compute.folds_routed() && let Some(at)=self.fold_waits(index,&plans[0],&all_values[0]) && at<steps_end {deferred.insert(at,index);continue;}
                    if self.expert_batch(index,span,&plans[0],&mut all_values[0])? {skip=span-1;continue;}
                }
                self.expert_rows(index,&plans,&mut all_values)?;continue;
            }
            for row in 0..count {
                self.import_row_state(layer,row,index,&plans,&names,state,&mut all_values)?;
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
                    if b.has_latent_absorption(&step.weights[0].name) {
                        let (query,output)=indirect::attention_tail(plan,index).ok_or("absorbed attention tail is not private")?;
                        // The count follows from the context. The Gather of this layer left a check of the device's count word
                        // (gather_absorbed), so no read waits here.
                        let layout=self.sparse_layout(layer)?;let live=selected_count(a.indexer.pool,a.indexer.token_budget,a.indexer.select_tail,context);
                        need(live<=step.outputs[0].shape[0].min(context),"absorbed selection count differs")?;
                        let live=live as usize;let query=self.local_query(values.get(query).ok_or("absorbed query is absent")?,1,layout)?;
                        let result=if live==0 {b.zeros(layout.local_heads()*layout.value_dim())?} else {
                            let latent=b.range(input[0],0,live*a.kv_rank as usize)?;
                            let s=Shape {op:Op::AbsorbQuery,heads:number(layout.local_heads() as u64)?,dim:number(a.key_dim)?,taps:number(a.value_dim)?,cols:number(a.kv_rank)?,rows:number(live as u64)?,epsilon:0.0,bound:0.0};
                            b.latent_absorption(&step.weights[0].name,s,&query,&latent,input[1],step.outputs[0].shape[0].min(self.capacity) as usize)?
                        };
                        values.insert(output.to_owned(),b.gather_attended(&result,1,layout)?);skip=3;continue;
                    }
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
                    if one_row {
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
                        TokenMixer::Kda(k)=>{s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;if kind==Kind::KdaPrepare {s.bound=k.gate_lower_bound as f32;}},
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
                    // The Gather's read of the ids makes the host wait for the device. The memo answers
                    // the expansion will ask for are made first, so that wait holds them.
                    if kind==Kind::Gather && count==1 && let Some(expand)=plan.steps.get(index+1).filter(|next|next.kind==Kind::ExpandKv && next.inputs.first()==step.outputs.first().map(|output|&output.name) && !next.weights.is_empty()) {
                        if !b.has_latent_absorption(&expand.weights[0].name) {b.prepare_origins(&expand.weights[0].name,input[0],s.dim as usize,true)?;}
                    }
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
                        // An absorbed layer needs neither the row identities nor the count on the host.
                        match (plan.steps.get(index+1).filter(|next|next.kind==Kind::ExpandKv && !next.weights.is_empty() && b.has_latent_absorption(&next.weights[0].name)),mixer) {
                            (Some(_),TokenMixer::LatentAttention(a))=>b.gather_absorbed(s,args,reserve,selected_count(a.indexer.pool,a.indexer.token_budget,a.indexer.select_tail,context) as f32)?,
                            _=>b.gather_reserved(s,args,reserve)?,
                        }
                    } else if kind==Kind::Scores {
                        let query=self.local_query(input[0],1,self.sparse_layout(layer)?)?;operation(s,[Some(&query),args[1],args[2],args[3]])?
                    } else {operation(s,args)?};
                    let result=if kind==Kind::Values {b.gather_attended(&result,1,self.sparse_layout(layer)?)?} else if kind==Kind::GatedNorm {b.gather_attended(&result,1,self.kda_layout(layer)?)?} else {result};
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
        let mut output=Vec::new();
        if let Some((_,held))=span {*held=all_values.remove(0);return Ok((output,BTreeMap::new()));}
        if !state_only {for values in &mut all_values {output.push(values.remove("output").ok_or("bounded layer has no output")?);}}
        let last=all_values.last_mut().ok_or("empty prompt layer")?;let mut next=BTreeMap::new();
        for (input,output) in names {next.insert(input.into(),last.remove(output).ok_or("bounded layer has no next state")?);}Ok((output,next))
    }
    fn import_row_state(&self,layer:usize,row:usize,index:usize,plans:&[&LayerPlan],names:&[(&str,&str)],state:&BTreeMap<String,C::Value>,values:&mut [BTreeMap<String,C::Value>])->Result<()> {
        for input in &plans[row].steps[index].inputs {
            if values[row].contains_key(input) || input=="local_slots" {continue;}
            let Some(spec)=plans[row].inputs.iter().find(|spec|&spec.name==input) else {continue;};
            let source=if row==0 {state.get(input).ok_or("bounded layer state input is absent")?} else {
                let output=names.iter().find(|(name,_)|*name==input).ok_or("prompt state output binding is absent")?.1;
                values[row-1].get(output).ok_or("prompt state predecessor has not completed")?
            };
            let value=self.compute.range(source,0,self.state_words(layer,spec.elements().map_err(|e|e.to_string())?)?)?;values[row].insert(input.clone(),value);
        }Ok(())
    }
    fn project_rows(&self,name:&str,input:&[&C::Value],round:bool)->Result<Vec<C::Value>> {
        if let Some(values)=self.compute.projection_rows(self.checkpoint,name,input,round)? {return Ok(values);}
        input.iter().map(|row|self.project(name,row,round)).collect()
    }
    /// LOCAL FOLD. The fold of the picked experts is row by row (one output word reads that word of every pick, in ID
    /// order, then the shared word), so the rank that owns a down row can fold it before the exchange with the bits
    /// the fold after the gather gives. When the fold is the only reader of the down outputs, the compute may run both
    /// steps as one stage and exchange the folded rows only. Some: the fold step's index, its output is stored.
    fn expert_fold(&self,index:usize,plan:&LayerPlan,values:&mut BTreeMap<String,C::Value>)->Result<Option<usize>> {
        let Some((at,fold))=fold_of(index,plan) else {return Ok(None);};let down=&plan.steps[index];
        let (Some(input),Some(ids),Some(weights))=(values.get(&down.inputs[0]),values.get(&fold.inputs[0]),values.get(&fold.inputs[1])) else {return Ok(None);};
        let shared=match fold.inputs.get(3) {Some(shared)=>{let Some(value)=values.get(shared) else {return Ok(None);};Some(value)},None=>None};
        let mut s=shape(Op::ExpertFold);
        s.rows=number(plan.steps.iter().find(|s|s.kind==Kind::Route).ok_or("bounded expert fold has no route")?.outputs[0].shape[0])?;
        s.cols=number(fold.outputs[0].shape[0])?;s.taps=u32::from(shared.is_some());
        let Some(output)=self.compute.routed_fold(std::slice::from_ref(down),ids,input,weights,shared,s)? else {return Ok(None);};
        values.insert(fold.outputs[0].name.clone(),output);Ok(Some(at))
    }
    /// The fold step of the routed down step `index` when the down step's own inputs and the fold's route are ready but
    /// another fold input is not (the shared expert's row runs after the routed down in the plan; an accepted earlier stage
    /// hoists it, a refused one does not). The steps between read neither the down outputs nor the fold output.
    fn fold_waits(&self,index:usize,plan:&LayerPlan,values:&BTreeMap<String,C::Value>)->Option<usize> {
        let (at,fold)=fold_of(index,plan)?;let down=&plan.steps[index];
        let ready=|name:&String|values.contains_key(name);
        (down.inputs.iter().all(ready) && fold.inputs[..2].iter().all(ready) && !fold.inputs[3..].iter().all(ready)).then_some(at)
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
pub(crate) use rows::{PromptChat,PromptResult,prompt_cohort_fits,prompt_cohort_growth};
#[path="bounded_decoder/batch.rs"] mod batch;
pub use batch::{BATCH_ROWS,BATCH_MIN_ROWS,BATCH_PAIR_ROWS,expert_union_counts,chat_attention_layers};
pub use rows::{absorbed_chat_words,expanded_row_words,group_workspace_bytes_with,LatentRows};
#[cfg(test)] pub(crate) use rows::wide_cut_layers;
pub use rows::{absorbed_group_words,full_memo_ballast,HeadRows,PROMPT_GROUP,PROMPT_GROUP_MAX,PROMPT_GROUP_LEASE,lease_groups,PROMPT_GROUP_CHECK,PROMPT_SHARED,ROW_STAGE_ROWS,RESERVED_ROWS,HELD_ROOM,POOL_ROOM,DECODE_RESERVE,DECODE_RETURNS,MEMO_KEEP,trims_spares_first,group_pool_cap,PROJECTION_MEMO_BYTES,expanded_bytes,group_workspace_bytes,group_workspace_bytes_for,OpenRows,SharedRows};
mod indirect;
#[path="bounded_decoder/projection_stage.rs"] mod projection_stage;
#[cfg(any(target_os="linux",test))] #[path="bounded_decoder/held.rs"] mod held;
#[path="bounded_decoder/proposer.rs"] mod proposer;
#[path="bounded_decoder/chains.rs"] mod chains;
pub use chains::ChainStart;
pub use proposer::{PROPOSER_HIDDEN,CHAIN_STOP,CHAIN_STOP_NAMED,PredictorState};
#[path="bounded_decoder/verify.rs"] mod verify;
pub use verify::{Verify,CARRIED_ROWS,VERIFY_ROWS,TRAIL_ROWS,PROMPT_TRAIL,SEED_PASS_ROWS,SEED_ROOM,learned_pass_ms,shared_pass,propose_group};
#[path="bounded_decoder/host.rs"] mod host;
pub use host::Host;
#[cfg(test)] pub(crate) use host::shadow_fixture::Checkpoint as TestCheckpoint;
#[cfg(target_os="linux")] #[path="bounded_decoder/device.rs"] mod device;
#[cfg(all(target_os="linux",feature="kernel-bench"))] #[path="bounded_decoder/projection_probe.rs"] mod projection_probe;
#[cfg(target_os="linux")] pub use device::{Device,Value,SLAB_ROWS};
#[cfg(target_os="linux")] #[path="bounded_decoder/factory.rs"] mod factory;
#[cfg(target_os="linux")] pub use factory::{Factory,ResidentFactory};
#[cfg(test)]
mod selection_read_tests {
    use engine_model::plan::description::PoolIndexerDescription;
    /// GLM's indexer (budget 2,048, pool 4, tail): the decode step knows its selection up to context 2,051 (2,048 + 3)
    /// and reads it back from 2,052 on; select_admission's own tests name the same bound.
    #[test] fn decode_steps_read_the_selection_only_past_the_known_bound() {
        let glm=PoolIndexerDescription {heads:32,dim:128,token_budget:2048,pool:4,learned_pool:false,select_tail:true,share_for_draft_iteration:false};
        for context in [1u64,2,15,128,1041,2048,2050,2051] {assert!(!super::selection_read(&glm,context),"context {context}");}
        for context in [2052u64,2053,4159,8193,32768] {assert!(super::selection_read(&glm,context),"context {context}");}
        assert!(super::selection_read(&PoolIndexerDescription {pool:0,..glm},16),"a shape it cannot form counts as a read");
    }
}
