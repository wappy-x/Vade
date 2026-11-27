//! The time profile of one token of the block family with a loop: where the time of a token step of the decode
//! driver (`spark_tp::loop_decoder::Driver`) goes on one backend.
//!
//! Mode `loop-token-time MODEL TRACE OUT NEW` (needs a GPU): the checkpoint of `MODEL` is loaded on one device.
//! The prompt of the trace folder (`prompt.ids`) runs, then `NEW` more tokens, greedy: each is fed the top row of
//! the token before it. No stream and no logit is read back. The chat runs three times from a clean state:
//! - `plain`: the driver on the backend as it is. The clock time of each token step.
//! - `calls`: every call of the driver goes through a backend that counts it by kind and adds up the clock time
//!   that the host spends inside the call. A call that only puts work in the queue of the device is short. A call
//!   that waits for the device is long. So this run shows where the host waits.
//! - `calls_with_waits`: the run `calls`, but each state write in place (`Compute::state_fill`) goes through
//!   `Compute::rows_fill`, which waits for the device first. The clock of `calls` against this run is the gain of
//!   the state write with no wait, both with the same cost of counting.
//! - `settled`: the same as `calls`, and after each call the host waits until the device has done all queued work. The time
//!   of a call with its wait is the time of that work alone on the device, with the cost of one wait in it. So
//!   this run shows the device time of each kind, one kind after the other. It is slower than the plain run.
//!
//! The kinds: an arm of the general kernel by its name, a row product by the name of its weight, `rows_fill` (a
//! state write in place), `upload`, `read`, `rows_greedy` (the top row), `range`, `join`, `zeros`, `round`.
//!
//! After the three runs, the floor of a launch: 4,863 launches of one kind that do almost no work, back to back
//! with one wait at the end, and the same with a wait after each launch. Three kinds: the arm `PackBf16` of the
//! general kernel on 2 values, the copy `round` of 1 word, and a row product on the smallest weight unit that
//! stays on the device. The time of a launch with one wait at the end is the cost of a launch in a queue.
//!
//! Mode `loop-rows-time MODEL FOLDER OUT ROWS POSITIONS NEW` (needs a GPU): the same three runs for a chat whose
//! prompt is read in steps of rows (`Driver::step_rows`). `FOLDER` holds `prompt.ids` and `tokens.tsv` (a trace
//! folder or a long reference). The first `POSITIONS` token IDs are the prompt, read in steps of `ROWS` rows; then
//! `NEW` tokens run, each as a step of one row, greedy. Lines `LOOP-ROWS-TIME`: the clock of the prompt (prompt
//! rows a second) and of a token after it; for the timed runs the kinds of the prompt for one ROW and the kinds
//! of a token after the prompt; the launches of the prompt (unit launches of an arm, wrapped launches, products);
//! and the bytes that the backend holds before and at the most. A wrapped launch is kind `wrapped` with the name
//! of its arm, another prompt operation is kind `prompt` with its name. The top row of the last prompt row must
//! be the one of `tokens.tsv`.
//!
//! The three runs must give the same top row at every position, or the mode fails. `tokens.tsv` of the trace is
//! compared too and the receipt says how many positions equal it. Receipt: `RESULT.txt`, and `COMPLETE` when the
//! three runs are equal. Mode `loop-token-time-host OUT` (any host) runs the same three runs on the made-up
//! checkpoints of the driver on the host backend: a check of this file, not a time of anything.
//! Every size comes from the description of the checkpoint and from the trace. No line here names a model.
use std::{cell::RefCell,collections::BTreeMap,fs,path::Path,time::{Duration,Instant}};
use engine_model::plan::{admission::CheckpointAdmission,description::LoopTextDescription};
use spark_gpu_probe::{causal_ops::Shape,norm_boundary::NormShape};
use spark_tp::{Result,bounded_decoder::{ChatGather,Compute,Host},layer_operands::ResidualBlockOperands,loop_decoder::{ChatStep,Driver,Wanted,fixture::{Fixture,SMALL,WIDE},mix::{Weight,Weights},rows::Limits,state},mxfp8_projection::Mxfp8Projection};

/// The calls of one kind: their count, the clock time inside the calls, the clock time of the waits after them.
#[derive(Default,Clone,Copy)]
/// `words`: the matrix words that the calls read (rows times columns of each weight), for a row product kind.
struct Cost {calls:u64,inside:Duration,wait:Duration,words:u64}
/// A backend that counts and times each call of the driver by kind and hands the call on to `inner` unchanged.
/// With `settle`, the host waits after each call until the device has done all queued work.
/// With `waits`, a state write in place goes through `rows_fill`.
pub struct Timed<'a,C:Compute> {inner:&'a C,settle:Option<&'a dyn Fn()->Result<()>>,costs:RefCell<BTreeMap<String,Cost>>,waits:bool}
impl<'a,C:Compute> Timed<'a,C> {
    fn new(inner:&'a C,settle:Option<&'a dyn Fn()->Result<()>>)->Self {Self {inner,settle,costs:RefCell::new(BTreeMap::new()),waits:false}}
    fn clear(&self) {self.costs.borrow_mut().clear();}
    fn timed<T>(&self,kind:impl FnOnce()->String,run:impl FnOnce()->Result<T>)->Result<T> {
        let started=Instant::now();let out=run();let inside=started.elapsed();
        let wait=match (self.settle,&out) {(Some(settle),Ok(_))=>{let started=Instant::now();settle()?;started.elapsed()},_=>Duration::ZERO};
        let mut costs=self.costs.borrow_mut();let cost=costs.entry(kind()).or_default();
        cost.calls+=1;cost.inside+=inside;cost.wait+=wait;out
    }
    /// `timed` for a row product: `words` gives the matrix words of the call from its output.
    fn sized<T>(&self,kind:impl FnOnce()->String,words:impl FnOnce(&T)->u64,run:impl FnOnce()->Result<T>)->Result<T> {
        let mut name=String::new();
        let out=self.timed(||{name=kind();name.clone()},run)?;
        if let Some(cost)=self.costs.borrow_mut().get_mut(&name) {cost.words+=words(&out);}
        Ok(out)
    }
}
impl<C:Compute> Compute for Timed<'_,C> {
    fn loop_prefill(&self, prefill: bool) -> bool { self.inner.loop_prefill(prefill) }
    type Value=C::Value;type Pass=C::Pass;
    fn upload_constant(&self,v:&[f32])->Result<C::Value> {self.timed(||format!("constant of {} words",v.len()),||self.inner.upload_constant(v))}
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {self.inner.work(run)}
    fn rows_small_pass(&self,rows:usize) {self.inner.rows_small_pass(rows)}
    fn mhc_row_launches(&self)->u64 {self.inner.mhc_row_launches()}
    fn rows_greedy(&self,logits:&C::Value,rows:usize)->Result<Option<Vec<u32>>> {self.timed(||"rows_greedy (the top row)".into(),||self.inner.rows_greedy(logits,rows))}
    fn has_latent_absorption(&self,name:&str)->bool {self.inner.has_latent_absorption(name)}
    fn gather_absorbed(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize,expected:f32)->Result<C::Value> {self.inner.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:Shape,query:&C::Value,latent:&C::Value,count:&C::Value,slots:usize)->Result<C::Value> {self.inner.latent_absorption(name,s,query,latent,count,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,query:&C::Value,latent:&C::Value,selected:&C::Value)->Result<C::Value> {self.inner.latent_absorption_rows(name,s,query,latent,selected)}
    fn latent_absorption_chats(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,gathers:&[ChatGather<'_,C::Value>],stride:usize,query:&C::Value,selected:&C::Value)->Result<Option<C::Value>> {
        self.inner.latent_absorption_chats(name,s,gathers,stride,query,selected)
    }
    fn upload(&self,v:&[f32])->Result<C::Value> {self.timed(||format!("upload of {} words",v.len()),||self.inner.upload(v))}
    fn read(&self,v:&C::Value)->Result<Vec<f32>> {self.timed(||format!("read of {} words",self.inner.words(v)),||self.inner.read(v))}
    fn select_known(&self,ids:&C::Value,count:&C::Value,row:&[f32])->Result<()> {self.inner.select_known(ids,count,row)}
    fn prefill_kv(&self, conv: &spark_tp::bounded_decoder::PrefillKv<'_, C::Value>, plane: &mut C::Value) -> Result<Option<C::Value>> {
        self.timed(||"prefill_kv".into(), ||self.inner.prefill_kv(conv, plane))
    }
    fn prompt_end(&self)->Result<()> {self.inner.prompt_end()}
    fn prompt_seeds(&self)->bool {self.inner.prompt_seeds()}
    fn prepare_beside(&self,c:&CheckpointAdmission,live:usize)->Result<()> {self.inner.prepare_beside(c,live)}
    fn words(&self,v:&C::Value)->usize {self.inner.words(v)}
    fn range(&self,v:&C::Value,start:usize,len:usize)->Result<C::Value> {self.timed(||"range".into(),||self.inner.range(v,start,len))}
    fn join(&self,parts:&[&C::Value])->Result<C::Value> {self.timed(||"join".into(),||self.inner.join(parts))}
    fn zeros(&self,len:usize)->Result<C::Value> {self.timed(||"zeros".into(),||self.inner.zeros(len))}
    fn operation(&self,s:Shape,x:[Option<&C::Value>;4],w:&[f32])->Result<C::Value> {self.timed(||format!("arm {:?}",s.op),||self.inner.operation(s,x,w))}
    fn norm(&self,name:&str,s:NormShape,x:&C::Value,w:&[f32])->Result<C::Value> {self.timed(||"norm kernel".into(),||self.inner.norm(name,s,x,w))}
    fn begin(&self,key:(usize,usize),p:&ResidualBlockOperands,x:&C::Value)->Result<(C::Pass,C::Value)> {self.inner.begin(key,p,x)}
    fn finish(&self,p:C::Pass,x:&C::Value)->Result<C::Value> {self.inner.finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&C::Value)->Result<C::Value> {self.inner.mx(p,x)}
    fn round(&self,x:&C::Value)->Result<C::Value> {self.timed(||"round".into(),||self.inner.round(x))}
    fn mean(&self,x:&C::Value,streams:u32,hidden:u32)->Result<C::Value> {self.inner.mean(x,streams,hidden)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&C::Value>;5],w:&[f32])->Result<C::Value> {
        self.timed(||if s.op==spark_gpu_probe::prompt_ops::Op::Wrapped {format!("wrapped {:?}",s.inner.op)} else {format!("prompt {:?}",s.op)},||self.inner.prompt(s,x,w))
    }
    fn rows_fill(&self,target:&mut C::Value,offset:usize,part:&C::Value,expected:usize)->Result<()> {
        self.timed(||"rows_fill (a state write in place)".into(),||self.inner.rows_fill(target,offset,part,expected))
    }
    fn state_fill(&self,target:&mut C::Value,offset:usize,part:&C::Value)->Result<()> {
        if self.waits {return self.timed(||"rows_fill (a state write in place)".into(),||self.inner.rows_fill(target,offset,part,self.inner.words(part)));}
        self.timed(||"state_fill (a state write in place, no wait)".into(),||self.inner.state_fill(target,offset,part))
    }
    fn state_restore(&self,live:&C::Value,full:usize)->Result<C::Value> {
        self.timed(||"state_restore".into(),||self.inner.state_restore(live,full))
    }
    fn dot_rows(&self,queries:&[f32],stack:&C::Value,count:usize,width:usize)->Result<C::Value> {
        self.timed(||"dot_rows (the dots of the block mix)".into(),||self.inner.dot_rows(queries,stack,count,width))
    }
    /// Rows of one block arm in one launch. A backend without the launch gives None and the caller runs one launch a
    /// row, each timed as its arm.
    fn block_rows(&self,s:Shape,rows:usize,shared:u32,x:[Option<&C::Value>;4],w:&[f32])->Result<Option<C::Value>> {
        self.timed(||format!("block_rows {:?}",s.op),||self.inner.block_rows(s,rows,shared,x,w))
    }
    #[allow(clippy::too_many_arguments)]
    fn mix_blend_transposed(&self,rows:usize,asked:usize,kept:usize,dim:usize,scale:f32,scores:&C::Value,joined:&C::Value)->Result<Option<C::Value>> {
        self.timed(||"mix_blend_transposed".into(),||self.inner.mix_blend_transposed(rows,asked,kept,dim,scale,scores,joined))
    }
    #[allow(clippy::too_many_arguments)]
    fn flash_window_attn(
        &self, rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
        scale: f32, first_position: u64, query: &C::Value, keys: &C::Value, values: &C::Value, moved: &C::Value,
        rel_table: &[f32], sink: &C::Value, gate: &C::Value, steps: &C::Value,
    ) -> Result<Option<C::Value>> {
        self.timed(|| "flash_window_attn".into(), || self.inner.flash_window_attn(rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, first_position, query, keys, values, moved, rel_table, sink, gate, steps))
    }
    fn window_one(&self,a:&spark_tp::bounded_decoder::WindowOne<'_,C::Value>)->Result<Option<C::Value>> {
        self.timed(|| "window_one".into(), || self.inner.window_one(a))
    }
    fn window_split(&self,a:&spark_tp::bounded_decoder::WindowSplit<'_,C::Value>)->Result<Option<C::Value>> {
        self.timed(|| "window_split".into(), || self.inner.window_split(a))
    }
    fn window_rows(&self,a:&spark_tp::bounded_decoder::WindowOne<'_,C::Value>,first_position:u64)->Result<Option<C::Value>> {
        self.timed(|| "window_rows".into(), || self.inner.window_rows(a,first_position))
    }
    fn conv_finish(&self,a:&spark_tp::bounded_decoder::ConvFinish<'_,C::Value>,ring:&mut C::Value,plane:&mut C::Value)->Result<Option<C::Value>> {
        self.timed(|| "conv_finish".into(), || self.inner.conv_finish(a,ring,plane))
    }
    fn mix_rows(&self,rows:usize,dim:usize,scale:f32,eps:f32,transpose:bool,keep_normed:bool,
        queries:&[f32],list:&[&C::Value],stream:&C::Value)->Result<Option<(Option<C::Value>,C::Value)>> {
        self.timed(|| "mix_rows".into(), ||self.inner.mix_rows(rows,dim,scale,eps,transpose,keep_normed,queries,list,stream))
    }
    #[allow(clippy::too_many_arguments)]
    fn mix_rows_normed(
        &self, rows: usize, dim: usize, scale: f32, eps: f32, post_eps: f32,
        transpose: bool, keep_normed: bool, keep_mixed: bool,
        queries: &[f32], list: &[&C::Value], stream: &C::Value,
        post_weight: Option<&[f32]>, post_weight_per_query: bool,
    ) -> Result<Option<(Option<C::Value>, Option<C::Value>, C::Value)>> {
        self.timed(|| "mix_rows_normed".into(), || self.inner.mix_rows_normed(rows, dim, scale, eps, post_eps, transpose, keep_normed, keep_mixed, queries, list, stream, post_weight, post_weight_per_query))
    }
    fn gate_blend_norm(
        &self, rows: usize, dim: usize, eps: f32,
        entries: &[(&C::Value, &C::Value, &C::Value, Option<&[f32]>)],
    ) -> Result<Option<Vec<C::Value>>> {
        self.timed(|| "gate_blend_norm".into(), || self.inner.gate_blend_norm(rows, dim, eps, entries))
    }
    fn mix_one(&self,queries:&[f32],list:&[&C::Value],x:&C::Value,width:usize,epsilon:f32,scale:f32)->Result<Option<(C::Value,C::Value)>> {
        self.timed(|| "mix_one".into(), || self.inner.mix_one(queries,list,x,width,epsilon,scale))
    }
    fn rows_norm(&self,name:&str,s:NormShape,x:&C::Value,m:usize,w:&[f32])->Result<C::Value> {self.inner.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&ResidualBlockOperands,x:&C::Value,m:usize)->Result<(C::Pass,C::Value)> {self.inner.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:C::Pass,x:&C::Value,m:usize)->Result<C::Value> {self.inner.rows_finish(p,x,m)}
}
/// The weights of `inner` for the timed backend: a row product is one timed call, by the name of its weight.
pub struct TimedWeights<'a,W> {inner:&'a W}
impl<C:Compute,W:Weights<C>> Weights<Timed<'_,C>> for TimedWeights<'_,W> {
    type RoutePlan = W::RoutePlan;
    fn arithmetic_phase(&self,decode:bool)->bool {self.inner.arithmetic_phase(decode)}
    fn words(&self,weight:Weight)->Result<&[f32]> {self.inner.words(weight)}
    /// The grouped products of `inner` (one launch for several matrices) are one timed call, by the name of the first weight.
    fn group_products(&self,compute:&Timed<'_,C>,weights:&[Weight],xs:&[&C::Value])->Result<Option<C::Value>> {
        compute.timed(||{let name=format!("{:?}",weights.first());format!("group_products {}",name.split('(').nth(1).unwrap_or(""))},||self.inner.group_products(compute.inner,weights,xs))
    }
    fn product(&self,compute:&Timed<'_,C>,weight:Weight,x:&C::Value)->Result<C::Value> {
        let columns=compute.words(x) as u64;
        compute.sized(||{let name=format!("{weight:?}");format!("product {}",name.split('(').next().unwrap_or(""))},|y|compute.words(y) as u64*columns,||self.inner.product(compute.inner,weight,x))
    }
    /// The products of `inner` on one input are one timed call, by the names of their weights.
    fn stack_products(&self,compute:&Timed<'_,C>,weights:&[Weight],x:&C::Value)->Result<Vec<C::Value>> {
        if weights.len()==1 {return Ok(vec![self.product(compute,weights[0],x)?]);}
        let columns=compute.words(x) as u64;
        compute.sized(||{let names=weights.iter().map(|weight|{let name=format!("{weight:?}");name.split('(').next().unwrap_or("").to_owned()}).collect::<Vec<_>>();format!("products {}",names.join("+"))},
            |ys:&Vec<C::Value>|ys.iter().map(|y|compute.words(y) as u64).sum::<u64>()*columns,||self.inner.stack_products(compute.inner,weights,x))
    }
    /// The grouped product of `inner` (the picked experts of one projection) is one timed call, so the time mode runs
    /// the launch the driver runs. Its matrix words are those of the picked matrices, whatever form the owner holds.
    fn products(&self,compute:&Timed<'_,C>,weights:&[Weight],x:&C::Value,shared:bool)->Result<Option<C::Value>> {
        let columns=compute.words(x) as u64/if shared {1} else {weights.len().max(1) as u64};
        compute.sized(||{let name=weights.first().map(|weight|format!("{weight:?}")).unwrap_or_default();format!("products {}",name.split('(').next().unwrap_or(""))},
            |y:&Option<C::Value>|y.as_ref().map_or(0,|y|compute.words(y) as u64*columns),||self.inner.products(compute.inner,weights,x,shared))
    }
    fn picked_gate(&self,compute:&Timed<'_,C>,layer:usize,ids:&C::Value,x:&C::Value,shared:bool,caps:[f32;4])->Result<Option<C::Value>> {
        let columns=compute.words(x) as u64/if shared {1} else {compute.words(ids).max(1) as u64};
        compute.sized(||"picked gate and activation".into(),|y:&Option<C::Value>|y.as_ref().map_or(0,|y|2*compute.words(y) as u64*columns),
            ||self.inner.picked_gate(compute.inner,layer,ids,x,shared,caps))
    }
    /// The picked products of `inner` with the numbers on the device are one timed call. Matrix words: the output
    /// words times the columns of one input row.
    fn picked_products(&self,compute:&Timed<'_,C>,first:Weight,ids:&C::Value,x:&C::Value,shared:bool,token_rows:usize)->Result<Option<C::Value>> {
        let (picks,inputs)=(compute.words(ids) as u64,compute.words(x) as u64);
        let columns=if shared {inputs} else {inputs/picks.max(1)};
        compute.sized(||{let name=format!("{first:?}");format!("picked {}",name.split('(').next().unwrap_or(""))},
            |y:&Option<C::Value>|y.as_ref().map_or(0,|y|compute.words(y) as u64*columns),||self.inner.picked_products(compute.inner,first,ids,x,shared,token_rows))
    }
    /// The product for rows of `inner` is one timed call, so an owner of weights with a launch for many rows is
    /// timed with that launch. With one row it is `product`.
    fn product_rows(&self,compute:&Timed<'_,C>,weight:Weight,x:&C::Value,rows:usize)->Result<C::Value> {
        if rows==1 {return self.product(compute,weight,x);}
        let columns=(compute.words(x)/rows) as u64;
        compute.sized(||{let name=format!("{weight:?}");format!("product_rows {}",name.split('(').next().unwrap_or(""))},|y|(compute.words(y)/rows) as u64*columns,||self.inner.product_rows(compute.inner,weight,x,rows))
    }
    fn stored_rows(&self,compute:&Timed<'_,C>,weight:Weight,tokens:&[u32])->Result<Option<C::Value>> {
        compute.timed(||format!("stored_rows {weight:?}"),||self.inner.stored_rows(compute.inner,weight,tokens))
    }
    fn stack_product_rows(&self,compute:&Timed<'_,C>,weights:&[Weight],x:&C::Value,rows:usize)->Result<Option<C::Value>> {
        let columns=(compute.words(x)/rows.max(1)) as u64;
        compute.sized(||{let names=weights.iter().map(|w|{let n=format!("{w:?}");n.split('(').next().unwrap_or("").to_owned()}).collect::<Vec<_>>();format!("stack_product_rows {}",names.join("+"))},
            |y:&Option<C::Value>|y.as_ref().map_or(0,|y|(compute.words(y)/rows.max(1)) as u64*columns),||self.inner.stack_product_rows(compute.inner,weights,x,rows))
    }
    fn stack_products_rows(&self,compute:&Timed<'_,C>,weights:&[Weight],x:&C::Value,rows:usize)->Result<Option<Vec<C::Value>>> {
        let columns=(compute.words(x)/rows.max(1)) as u64;
        compute.sized(||{let names=weights.iter().map(|w|{let n=format!("{w:?}");n.split('(').next().unwrap_or("").to_owned()}).collect::<Vec<_>>();format!("stack_products_rows {}",names.join("+"))},
            |ys:&Option<Vec<C::Value>>|ys.as_ref().map_or(0,|ys|ys.iter().map(|y|(compute.words(y)/rows.max(1)) as u64).sum::<u64>()*columns),||self.inner.stack_products_rows(compute.inner,weights,x,rows))
    }
    fn product_rows_pair(&self,compute:&Timed<'_,C>,weight:Weight,x1:&C::Value,x2:&C::Value,rows:usize)->Result<Option<C::Value>> {
        let columns=(compute.words(x1)*2/rows.max(1)) as u64;
        compute.sized(||{let name=format!("{weight:?}");format!("product_rows_pair {}",name.split('(').next().unwrap_or(""))},
            |y:&Option<C::Value>|y.as_ref().map_or(0,|y|(compute.words(y)/rows.max(1)) as u64*columns),||self.inner.product_rows_pair(compute.inner,weight,x1,x2,rows))
    }
    fn device_route_group(&self,compute:&Timed<'_,C>,layer:usize,route:&C::Value,rows:usize,picks:usize,experts:usize)->Result<Option<(C::Value,Self::RoutePlan)>> {
        compute.timed(||"device_route_group".into(),||self.inner.device_route_group(compute.inner,layer,route,rows,picks,experts))
    }
    fn device_routed_products(&self,compute:&Timed<'_,C>,first:Weight,route:&Self::RoutePlan,x:&C::Value)->Result<Option<C::Value>> {
        compute.timed(||{let name=format!("{first:?}");format!("device_routed_products {}",name.split('(').next().unwrap_or(""))},
            ||self.inner.device_routed_products(compute.inner,first,route,x))
    }
    fn device_routed_gate(&self,compute:&Timed<'_,C>,layer:usize,route:&Self::RoutePlan,x:&C::Value,caps:[f32;4])->Result<Option<C::Value>> {
        compute.timed(||"device_routed_gate".into(),||self.inner.device_routed_gate(compute.inner,layer,route,x,caps))
    }
    fn product_runs(&self,compute:&Timed<'_,C>,runs:&[(Weight,usize)],x:&C::Value,columns:usize,input:Option<&[usize]>,output:Option<&[usize]>)->Result<C::Value> {
        compute.timed(||{let name=runs.first().map(|(w,_)|format!("{w:?}")).unwrap_or_default();format!("product_runs {}",name.split('(').next().unwrap_or(""))},
            ||self.inner.product_runs(compute.inner,runs,x,columns,input,output))
    }
}

/// One chat from a clean state: the prompt, then `new` tokens, each fed the top row before it. No read-back.
/// `after_prompt` runs before the first token after the prompt. Gives the top row and the clock time of each step.
fn chat<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,prompt:&[u32],new:usize,mut after_prompt:impl FnMut())->Result<(Vec<u32>,Vec<Duration>)> {
    let driver=Driver::new(c,w,d,(prompt.len()+new) as u64)?;let mut state=driver.clean()?;
    let (mut tops,mut times,mut token)=(Vec::new(),Vec::new(),0);
    for position in 0..prompt.len()+new {
        if position==prompt.len() {after_prompt();}
        if let Some(&id)=prompt.get(position) {token=id;}
        let started=Instant::now();
        let step=c.work(||driver.step(&mut state,token)).map_err(|e|format!("loop token time: position {position}, token {token}: {e}"))?;
        times.push(started.elapsed());tops.push(step.top);token=step.top;
    }
    Ok((tops,times))
}
fn ms(time:Duration,tokens:usize)->f64 {time.as_secs_f64()*1.0e3/tokens.max(1) as f64}
/// One line for the clock times of a run: the prompt apart, then the tokens after it.
fn clock(run:&str,times:&[Duration],prompt:usize)->String {
    let (before,after)=times.split_at(prompt.min(times.len()));let total:Duration=after.iter().sum();let each=ms(total,after.len());
    let part=|tokens:&[Duration]|ms(tokens.iter().sum(),tokens.len());let edge=after.len().min(8);
    format!("LOOP-TOKEN-TIME CLOCK run={run} prompt_tokens={} prompt_seconds={:.3} new_tokens={} new_seconds={:.3} ms_a_token={each:.3} tokens_a_second={:.3} first_{edge}_ms={:.3} last_{edge}_ms={:.3} least_ms={:.3} most_ms={:.3}",
        before.len(),before.iter().sum::<Duration>().as_secs_f64(),after.len(),total.as_secs_f64(),if each>0.0 {1.0e3/each} else {0.0},part(&after[..edge]),part(&after[after.len()-edge..]),
        after.iter().min().map_or(0.0,|t|ms(*t,1)),after.iter().max().map_or(0.0,|t|ms(*t,1)))
}
/// The lines of the kinds of one timed run, the largest time first, for the tokens after the prompt.
fn kinds(head:&str,unit:&str,run:&str,costs:&BTreeMap<String,Cost>,new:usize,token_ms:f64,lines:&mut Vec<String>) {
    let mut rows=costs.iter().map(|(kind,cost)|(kind.as_str(),*cost)).collect::<Vec<_>>();
    rows.sort_by_key(|row|std::cmp::Reverse(row.1.inside+row.1.wait));
    let (mut calls,mut inside,mut wait)=(0,Duration::ZERO,Duration::ZERO);
    for (kind,cost) in rows {
        calls+=cost.calls;inside+=cost.inside;wait+=cost.wait;
        // A row product kind also gives its matrix words a unit and the words a second of its calls with their waits.
        let words=if cost.words>0 {format!(" matrix_words_a_{unit}={} giga_words_a_second={:.1}",cost.words/new.max(1) as u64,cost.words as f64/(cost.inside+cost.wait).as_secs_f64().max(1.0e-12)/1.0e9)} else {String::new()};
        lines.push(format!("{head} KIND run={run} kind=\"{kind}\" calls_a_{unit}={:.2} ms_inside_calls={:.3} ms_waits_after={:.3} ms_a_{unit}={:.3} microseconds_a_call={:.1}{words}",
            cost.calls as f64/new.max(1) as f64,ms(cost.inside,new),ms(cost.wait,new),ms(cost.inside+cost.wait,new),(cost.inside+cost.wait).as_secs_f64()*1.0e6/cost.calls.max(1) as f64));
    }
    let all=ms(inside+wait,new);
    lines.push(format!("{head} KINDS run={run} calls_a_{unit}={:.2} ms_inside_calls={:.3} ms_waits_after={:.3} ms_in_all_kinds={all:.3} ms_a_{unit}_clock={token_ms:.3} ms_outside_the_calls={:.3}",
        calls as f64/new.max(1) as f64,ms(inside,new),ms(wait,new),token_ms-all));
}
/// The names of the counters that `counters` gives. A time is in nanoseconds.
const COUNTERS:[&str;7]=["queued_calls","stream_wait_calls","stream_wait_ns","device_to_host_copies","host_launch_ns","output_download_ns","allocation_ns"];
/// The three runs of one chat on `c`. `settle` waits until the backend has done all queued work. `counters` reads
/// the counters of the backend in the order of `COUNTERS`. Gives the lines of the receipt and whether the three
/// runs gave the same top rows.
pub fn profile<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,prompt:&[u32],new:usize,settle:&dyn Fn()->Result<()>,counters:&dyn Fn()->[u128;7],trace_tops:&[u32])->Result<(Vec<String>,bool)> {
    if prompt.is_empty() || new==0 {return Err("loop token time: the chat needs a prompt and at least one new token".into());}
    let mut lines=Vec::new();
    // The plain run: no backend of this file is in the way. The counters of the backend are read around the
    // tokens after the prompt.
    let mut before=[0;7];
    let (plain,times)=chat(c,w,d,prompt,new,||{before=counters();})?;
    settle()?;let after=counters();
    let after_prompt=|times:&[Duration]|ms(times[prompt.len()..].iter().sum(),new);
    let plain_ms=after_prompt(&times);
    lines.push(clock("plain",&times,prompt.len()));
    lines.push(format!("LOOP-TOKEN-TIME BACKEND run=plain for one token after the prompt: {}",COUNTERS.iter().zip(after.iter().zip(&before)).map(|(name,(after,before))|{
        let each=after.saturating_sub(*before) as f64/new as f64;
        if name.ends_with("_ns") {format!("{}_ms={:.3}",name.trim_end_matches("_ns"),each/1.0e6)} else {format!("{name}={each:.2}")}
    }).collect::<Vec<_>>().join(" ")));
    let mut equal=true;
    for (run,settle,waits) in [("calls",None,false),("calls_with_waits",None,true),("settled",Some(settle),false)] {
        let (timed,weights)=(Timed {waits,..Timed::new(c,settle)},TimedWeights {inner:w});
        let (tops,times)=chat(&timed,&weights,d,prompt,new,||timed.clear())?;
        lines.push(clock(run,&times,prompt.len()));
        kinds("LOOP-TOKEN-TIME","token",run,&timed.costs.borrow(),new,after_prompt(&times),&mut lines);
        let same=tops==plain;equal&=same;
        lines.push(format!("LOOP-TOKEN-TIME TOPS run={run} equal_to_plain_run={same} positions={}",tops.len()));
    }
    let seen=plain.len().min(trace_tops.len());let same=plain.iter().zip(trace_tops).filter(|(a,b)|a==b).count();
    let ids=|tops:&[u32]|tops.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
    lines.push(format!("LOOP-TOKEN-TIME TRACE positions_of_the_trace={seen} equal_to_plain_run={same}"));
    lines.push(format!("LOOP-TOKEN-TIME RESULT ms_a_token={plain_ms:.3} tokens_a_second={:.3} prompt_tokens={} new_tokens={new} visits={} runs_equal={equal} new_token_ids={}",
        if plain_ms>0.0 {1.0e3/plain_ms} else {0.0},prompt.len(),d.visits.len(),ids(&plain[prompt.len()-1..])));
    Ok((lines,equal))
}
/// One chat whose prompt is read in steps of rows: `prompt` in steps of `limits.step_rows` rows, of which only the
/// last row runs the end, then `new` tokens, each as a step of one row and fed the top row before it. No
/// read-back. `after_prompt` runs when the top row of the last prompt row is on the host. Gives the top rows from
/// the last prompt row on, the clock time of the prompt and the clock time of each token after it.
fn chat_rows<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,prompt:&[u32],new:usize,limits:Limits,mut after_prompt:impl FnMut())->Result<(Vec<u32>,Duration,Vec<Duration>)> {
    let driver=Driver::with_limits(c,w,d,(prompt.len()+new) as u64,limits)?;let mut state=driver.clean()?;
    let (rows,started,mut top)=(limits.step_rows.max(1),Instant::now(),None);
    for (number,tokens) in prompt.chunks(rows).enumerate() {
        let end=(number+1)*rows>=prompt.len();
        let step=c.work(||driver.step_rows(&mut [ChatStep {state:&mut state,tokens,wanted:if end {Wanted::Last} else {Wanted::None},logits:false}]))
            .map_err(|e|format!("loop rows time: step {number} of the prompt: {e}"))?;
        if end {top=step.tops.first().copied();}
    }
    let read=started.elapsed();
    let mut token=top.ok_or("loop rows time: the prompt gave no top row")?;
    after_prompt();
    let (mut tops,mut times)=(vec![token],Vec::new());
    for number in 0..new {
        let started=Instant::now();
        let step=c.work(||driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&[token],wanted:Wanted::Last,logits:false}]))
            .map_err(|e|format!("loop rows time: token {number} after the prompt: {e}"))?;
        token=step.tops.first().copied().ok_or("loop rows time: a token step gave no top row")?;
        times.push(started.elapsed());tops.push(token);
    }
    Ok((tops,read,times))
}
/// The three runs of one chat whose prompt is read in steps of rows. `settle` waits until the backend has done
/// all queued work. `memory` reads the bytes that the backend holds now and the most it held. `want` is the top
/// row that the last prompt row must give. Gives the lines of the receipt and whether the runs are equal.
pub fn profile_rows<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,prompt:&[u32],new:usize,limits:Limits,settle:&dyn Fn()->Result<()>,memory:&dyn Fn()->[usize;2],want:Option<u32>)->Result<(Vec<String>,bool)> {
    if prompt.is_empty() || new==0 {return Err("loop rows time: the chat needs a prompt and at least one new token".into());}
    const HEAD:&str="LOOP-ROWS-TIME";
    let clock=|run:&str,read:Duration,times:&[Duration]| {
        let each=ms(times.iter().sum(),times.len());
        format!("{HEAD} CLOCK run={run} rows_a_step={} prompt_rows={} prompt_seconds={:.4} prompt_rows_a_second={:.3} new_tokens={} ms_a_token={each:.3} tokens_a_second={:.3} least_ms={:.3} most_ms={:.3}",
            limits.step_rows,prompt.len(),read.as_secs_f64(),prompt.len() as f64/read.as_secs_f64().max(1.0e-9),times.len(),if each>0.0 {1.0e3/each} else {0.0},
            times.iter().min().map_or(0.0,|t|ms(*t,1)),times.iter().max().map_or(0.0,|t|ms(*t,1)))
    };
    let mut lines=Vec::new();
    settle()?;let before=memory();
    let (plain,read,times)=chat_rows(c,w,d,prompt,new,limits,||{})?;
    settle()?;let after=memory();
    lines.push(clock("plain",read,&times));
    lines.push(format!("{HEAD} MEMORY run=plain bytes_before={} most_bytes_before={} bytes_after={} most_bytes_after={} bound_of_a_step_bytes={}",before[0],before[1],after[0],after[1],
        Driver::with_limits(c,w,d,(prompt.len()+new) as u64,limits)?.transient_words(limits.step_rows.min(prompt.len()),1,1,0).saturating_mul(4)));
    let mut equal=want.is_none_or(|top|plain[0]==top);
    lines.push(format!("{HEAD} TOP of_the_last_prompt_row={} wanted={} equal={equal}",plain[0],want.map_or("none".to_owned(),|top|top.to_string())));
    for (run,settle) in [("calls",None),("settled",Some(settle))] {
        let (timed,weights)=(Timed::new(c,settle),TimedWeights {inner:w});
        let mut of_prompt=BTreeMap::new();
        let (tops,read,times)=chat_rows(&timed,&weights,d,prompt,new,limits,||{of_prompt=timed.costs.borrow().clone();timed.clear();})?;
        lines.push(clock(run,read,&times));
        kinds(&format!("{HEAD} PROMPT"),"row",run,&of_prompt,prompt.len(),ms(read,prompt.len()),&mut lines);
        let count=|head:&str|of_prompt.iter().filter(|(kind,_)|kind.starts_with(head)).map(|(_,cost)|cost.calls).sum::<u64>();
        lines.push(format!("{HEAD} LAUNCHES run={run} part=prompt prompt_rows={} steps={} unit_arm_launches={} wrapped_launches={} other_prompt_operations={} product_calls={} product_rows_calls={}",
            prompt.len(),prompt.len().div_ceil(limits.step_rows.max(1)),count("arm "),count("wrapped "),count("prompt "),count("product "),count("product_rows ")));
        kinds(&format!("{HEAD} TOKEN"),"token",run,&timed.costs.borrow(),new,ms(times.iter().sum(),new),&mut lines);
        let same=tops==plain;equal&=same;
        lines.push(format!("{HEAD} TOPS run={run} equal_to_plain_run={same} positions={}",tops.len()));
    }
    let each=ms(times.iter().sum(),new);
    lines.push(format!("{HEAD} RESULT rows_a_step={} prompt_rows={} prompt_rows_a_second={:.3} ms_a_token_after={each:.3} tokens_a_second_after={:.3} visits={} runs_equal={equal}",
        limits.step_rows,prompt.len(),prompt.len() as f64/read.as_secs_f64().max(1.0e-9),if each>0.0 {1.0e3/each} else {0.0},d.visits.len()));
    Ok((lines,equal))
}
/// Rows a step of the prompt of mode `loop-rows-price`.
const PRICE_PROMPT_STEP:usize=1024;
/// A copy of the rings of each visit of `state`: views, no word moves on a device.
fn rings_of<C:Compute>(c:&C,state:&state::State<C::Value>)->Result<Vec<[C::Value;4]>> {
    state.visits.iter().map(|visit|{let copy=|i:usize|c.range(&visit.rings[i],0,c.words(&visit.rings[i]));Ok([copy(0)?,copy(1)?,copy(2)?,copy(3)?])}).collect()
}
/// The state as it was at `position` with `rings`. A step of rows writes the rows of its own positions and moves the
/// rings and the position only: a row that a later step of the same positions reads is written again by it first, and
/// a ring slot it wrote over is out of every window from `position` on.
fn set_back<C:Compute>(c:&C,state:&mut state::State<C::Value>,position:u64,rings:&[[C::Value;4]])->Result<()> {
    need_price(state.visits.len()==rings.len(),"the saved rings differ from the visits")?;
    state.position=position;
    for (visit,saved) in state.visits.iter_mut().zip(rings) {for (ring,saved) in visit.rings.iter_mut().zip(saved) {*ring=c.range(saved,0,c.words(saved))?;}}
    Ok(())
}
fn need_price(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(format!("loop rows price: {why}"))}}
/// One step of `tokens` for one chat, every row wanted. The clock ends when the top rows are on the host.
fn price_step<C:Compute,W:Weights<C>>(c:&C,driver:&Driver<'_,C,W>,state:&mut state::State<C::Value>,tokens:&[u32])->Result<(Vec<u32>,Duration)> {
    let started=Instant::now();
    let step=c.work(||driver.step_rows(&mut [ChatStep {state,tokens,wanted:Wanted::All,logits:false}]))?;
    Ok((step.tops,started.elapsed()))
}
/// Mode `loop-rows-price`: for each prefix, one chat reads the first `prefix` IDs in steps of `PRICE_PROMPT_STEP`
/// rows; then each count R of `rows` runs as one step of the next R IDs, `repeats` times from the same state. Then,
/// after the largest prefix, the profile of one step of each count of `profile`: the backend counters around a plain
/// step, and the kinds of a step in the runs `calls` and `settled`. `tops` are the top rows of the reference by
/// position. Gives the lines of the receipt and whether every repeat and every timed run gave the same top rows.
#[allow(clippy::too_many_arguments)]
pub fn price_rows<C:Compute,W:Weights<C>>(c:&C,w:&W,d:&LoopTextDescription,ids:&[u32],tops:&[u32],prefixes:&[usize],rows:&[usize],profile:&[usize],repeats:usize,
    settle:&dyn Fn()->Result<()>,counters:&dyn Fn()->[u128;7])->Result<(Vec<String>,bool)> {
    const HEAD:&str="LOOP-ROWS-PRICE";
    let (most,last)=(rows.iter().chain(profile).copied().max().unwrap_or(0),prefixes.iter().copied().max().unwrap_or(0));
    need_price(most>0 && repeats>0 && !prefixes.is_empty() && rows.iter().chain(profile).all(|&r|r>0) && last+most<=ids.len(),"the prefixes and counts need IDs past the largest prefix")?;
    let (limits,capacity)=(Limits {step_rows:most.max(PRICE_PROMPT_STEP),..Limits::ONE},(last+most) as u64);
    let driver=Driver::with_limits(c,w,d,capacity,limits)?;
    let (mut lines,mut equal)=(vec![format!("{HEAD} PLAN prefixes={prefixes:?} rows={rows:?} profile={profile:?} repeats={repeats} step_row_limit={} capacity={capacity} wanted=all_rows",limits.step_rows)],true);
    let same_as=|got:&[u32],first:usize|got.iter().zip(tops.get(first..).unwrap_or(&[])).filter(|(a,b)|a==b).count();
    for &prefix in prefixes {
        let mut state=driver.clean()?;
        let started=Instant::now();
        for (number,tokens) in ids[..prefix].chunks(PRICE_PROMPT_STEP).enumerate() {
            c.work(||driver.step_rows(&mut [ChatStep {state:&mut state,tokens,wanted:Wanted::None,logits:false}])).map_err(|e|format!("loop rows price: prefix {prefix}, step {number}: {e}"))?;
        }
        settle()?;
        lines.push(format!("{HEAD} PREFIX prompt_rows={prefix} seconds={:.3}",started.elapsed().as_secs_f64()));
        let saved=rings_of(c,&state)?;
        let mut first_ms=None;
        for &r in rows {
            let tokens=&ids[prefix..prefix+r];
            let (mut times,mut seen)=(Vec::with_capacity(repeats),None::<Vec<u32>>);
            for _ in 0..repeats {
                set_back(c,&mut state,prefix as u64,&saved)?;
                let (got,time)=price_step(c,&driver,&mut state,tokens).map_err(|e|format!("loop rows price: prefix {prefix}, {r} rows: {e}"))?;
                times.push(time);
                match &seen {None=>seen=Some(got),Some(before)=>equal&=*before==got}
            }
            times.sort();
            let (least,middle)=(ms(times[0],1),ms(times[times.len()/2],1));
            let base=*first_ms.get_or_insert(middle);
            lines.push(format!("{HEAD} STEP prefix={prefix} rows={r} repeats={repeats} least_ms={least:.3} middle_ms={middle:.3} most_ms={:.3} r={:.3} rows_a_second={:.2} tops_equal_to_reference={} of={r}",
                ms(times[times.len()-1],1),middle/base,r as f64*1.0e3/middle.max(1.0e-9),same_as(seen.as_deref().unwrap_or(&[]),prefix)));
        }
        if prefix!=last {continue;}
        for &r in profile {
            let tokens=&ids[prefix..prefix+r];
            set_back(c,&mut state,prefix as u64,&saved)?;settle()?;let before=counters();
            let (plain,time)=price_step(c,&driver,&mut state,tokens)?;
            settle()?;let after=counters();
            lines.push(format!("{HEAD} PROFILE rows={r} prefix={prefix} run=plain ms_a_step={:.3} backend: {}",ms(time,1),COUNTERS.iter().zip(after.iter().zip(&before)).map(|(name,(after,before))|{
                let each=after.saturating_sub(*before) as f64;
                if name.ends_with("_ns") {format!("{}_ms={:.3}",name.trim_end_matches("_ns"),each/1.0e6)} else {format!("{name}={each:.0}")}
            }).collect::<Vec<_>>().join(" ")));
            for (run,settle) in [("calls",None),("settled",Some(settle))] {
                let (timed,weights)=(Timed::new(c,settle),TimedWeights {inner:w});
                let driver=Driver::with_limits(&timed,&weights,d,capacity,limits)?;
                set_back(c,&mut state,prefix as u64,&saved)?;timed.clear();
                let (got,time)=price_step(&timed,&driver,&mut state,tokens)?;
                kinds(&format!("{HEAD} PROFILE rows={r}"),"step",run,&timed.costs.borrow(),1,ms(time,1),&mut lines);
                let same=got==plain;equal&=same;
                lines.push(format!("{HEAD} PROFILE rows={r} run={run} ms_a_step={:.3} tops_equal_to_plain_run={same}",ms(time,1)));
            }
        }
    }
    lines.push(format!("{HEAD} RESULT runs_equal={equal} visits={}",d.visits.len()));
    Ok((lines,equal))
}
/// The count of launches of a floor run: the calls of one token step in the first profile of this driver.
const FLOOR_LAUNCHES:usize=4863;
/// `FLOOR_LAUNCHES` launches of one kind, twice: back to back with one wait at the end, then with a wait after
/// each launch. One launch and a wait run first, outside the clock. The line gives microseconds a launch.
fn floor<C:Compute>(c:&C,kind:&str,settle:&dyn Fn()->Result<()>,launch:&dyn Fn()->Result<C::Value>)->Result<String> {
    c.work(|| {launch()?;settle()})?;
    let queued=c.work(|| {let started=Instant::now();for _ in 0..FLOOR_LAUNCHES {launch()?;}settle()?;Ok(started.elapsed())})?;
    let settled=c.work(|| {let started=Instant::now();for _ in 0..FLOOR_LAUNCHES {launch()?;settle()?;}Ok(started.elapsed())})?;
    let each=|time:Duration|time.as_secs_f64()*1.0e6/FLOOR_LAUNCHES as f64;
    Ok(format!("LOOP-TOKEN-TIME FLOOR kind=\"{kind}\" launches={FLOOR_LAUNCHES} microseconds_a_launch_one_wait_at_the_end={:.3} microseconds_a_launch_wait_after_each={:.3}",each(queued),each(settled)))
}
/// The floor of a launch for three kinds that do almost no work. `unit` is the weight of the row product with its
/// rows and columns; None leaves the product out.
pub fn floors<C:Compute,W:Weights<C>>(c:&C,w:&W,unit:Option<(Weight,usize,usize)>,settle:&dyn Fn()->Result<()>)->Result<Vec<String>> {
    let (pair,one)=(c.upload(&[1.0,2.0])?,c.upload(&[1.0])?);
    let mut lines=vec![floor(c,"arm PackBf16, 2 values to 1 word",settle,&||state::pack(c,&pair))?,floor(c,"round of 1 word",settle,&||c.round(&one))?];
    if let Some((weight,rows,columns))=unit {
        let x=c.upload(&(0..columns).map(|column|(column%7) as f32*0.25).collect::<Vec<_>>())?;
        lines.push(floor(c,&format!("product {weight:?}, {rows} rows of {columns} columns"),settle,&||w.product(c,weight,&x))?);
    }
    Ok(lines)
}
/// Writes the receipt. Runs that are not equal are an error of the mode, after the receipt is written.
fn receipt(out:&Path,lines:&[String],equal:bool)->Result<()> {
    let write=|name:&str,text:String|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
    for line in lines {eprintln!("{line}");}
    write("RESULT.txt",lines.iter().map(|line|format!("{line}\n")).collect())?;
    if !equal {return Err("loop token time: a timed run gave another top row than the plain run; see RESULT.txt".into());}
    write("COMPLETE",String::new())
}
/// Mode `loop-token-time-host`: the three runs on the made-up checkpoints on the host backend.
pub fn fixtures(out:&Path)->Result<()> {
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut lines,mut equal)=(Vec::new(),true);
    for (name,config) in [("SMALL",SMALL),("WIDE",WIDE)] {
        let fixture=Fixture::new(config)?;let prompt=fixture.chat(4);
        lines.push(format!("LOOP-TOKEN-TIME PLAN checkpoint={name} backend=host prompt_tokens={} new_tokens=8 visits={}",prompt.len(),fixture.description.visits.len()));
        let (mut part,same)=profile(&Host,&fixture,&fixture.description,&prompt,8,&||Ok(()),&||[0;7],&[])?;
        lines.append(&mut part);equal&=same;
        lines.append(&mut floors(&Host,&fixture,Some((Weight::Head,fixture.description.head_rows as usize,fixture.description.hidden as usize)),&||Ok(()))?);
        let (mut part,same)=profile_rows(&Host,&fixture,&fixture.description,&fixture.chat(8),4,Limits {step_rows:3,..Limits::ONE},&||Ok(()),&||[0,0],None)?;
        lines.append(&mut part);equal&=same;
        let (mut part,same)=price_rows(&Host,&fixture,&fixture.description,&fixture.chat(12),&[],&[2,5],&[1,2,3],&[3],2,&||Ok(()),&||[0;7])?;
        lines.append(&mut part);equal&=same;
    }
    receipt(out,&lines,equal)
}
/// The token IDs of the prompt and the top row of each position of a trace folder.
#[cfg(target_os="linux")]
fn trace_ids(folder:&Path)->Result<(Vec<u32>,Vec<u32>)> {
    let text=|name:&str|fs::read_to_string(folder.join(name)).map_err(|e|format!("{}: {e}",folder.join(name).display()));
    let id=|file:&str,word:&str|word.trim().parse::<u32>().map_err(|_|format!("{file}: {word} is no token ID"));
    let prompt=text("prompt.ids")?.lines().map(|line|id("prompt.ids",line)).collect::<Result<Vec<_>>>()?;
    let tops=text("tokens.tsv")?.lines().map(|line|id("tokens.tsv",line.split_once('\t').map_or(line,|(_,top)|top))).collect::<Result<Vec<_>>>()?;
    Ok((prompt,tops))
}
/// Mode `loop-rows-time`: the real checkpoint on one device, the prompt read in steps of rows.
#[cfg(target_os="linux")]
pub fn run_rows(model:&Path,folder:&Path,out:&Path,rows:&str,positions:&str,new:&str)->Result<()> {rows_with(model,folder,out,rows,positions,new,false)}
/// Mode `loop-rows-time-pooled`: `loop-rows-time` with the vector pool of the backend on (margin 1 GiB), as a server
/// runs it. A word of the check program, no environment read.
#[cfg(target_os="linux")]
pub fn run_rows_pooled(model:&Path,folder:&Path,out:&Path,rows:&str,positions:&str,new:&str)->Result<()> {rows_with(model,folder,out,rows,positions,new,true)}
#[cfg(target_os="linux")]
fn rows_with(model:&Path,folder:&Path,out:&Path,rows:&str,positions:&str,new:&str,pool:bool)->Result<()> {
    let count=|name:&str,word:&str|word.parse::<usize>().ok().filter(|&n|n>0).ok_or_else(||format!("loop rows time: {name} must be a count above zero, got {word}"));
    let (rows,positions)=(count("ROWS",rows)?,count("POSITIONS",positions)?);
    run_with(model,folder,out,new,Some((Limits {step_rows:rows,..Limits::ONE},positions)),pool)
}
/// Mode `loop-rows-price`: the real checkpoint on one device. `FOLDER` holds `prompt.ids` and `tokens.tsv`.
#[cfg(target_os="linux")]
pub fn run_price(model:&Path,folder:&Path,out:&Path,prefixes:&str,rows:&str,profile:&str,repeats:&str)->Result<()> {
    use engine_model::plan::{loop_load,loop_state::{LoopLedger,LoopStatePlan,context_limit}};
    use spark_tp::{bounded_decoder::Device,loop_decoder::operands::{device::DeviceStore,load,weights::LoadedWeights}};
    let list=|name:&str,words:&str|words.split(',').map(|word|word.parse::<usize>().ok().filter(|&n|n>0).ok_or_else(||format!("loop rows price: {name} must be counts above zero, got {words}"))).collect::<Result<Vec<_>>>();
    let (prefixes,rows,profile,repeats)=(list("PREFIXES",prefixes)?,list("ROWS",rows)?,list("PROFILE",profile)?,list("REPEATS",repeats)?[0]);
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (ids,tops)=trace_ids(folder)?;
    let most=rows.iter().chain(&profile).copied().max().unwrap_or(0);
    let capacity=(prefixes.iter().copied().max().unwrap_or(0)+most) as u64;
    let checkpoint=loop_load::read_folder(model).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    let d=&checkpoint.description;
    // The memory rule of the load check, as in mode `loop-token-time`.
    let bucket=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
    let state=LoopStatePlan::new(d,capacity,bucket).map_err(|e|e.to_string())?;
    let ledger=LoopLedger::new(&checkpoint.load.bytes,&state).map_err(|e|e.to_string())?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let budget=free.saturating_sub((free/10).max(2<<30)) as u64;
    let limit=context_limit(d,&checkpoint.load.bytes,bucket,budget).map_err(|e|e.to_string())?;
    ledger.admit(budget,limit).map_err(|refusal|format!("text service memory refusal: {refusal}"))?;
    let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
    let mut lines=vec![format!("LOOP-ROWS-PRICE DEVICE ids={} visits={} width={} capacity={capacity} ledger_bytes={} budget_bytes={budget}",ids.len(),d.visits.len(),d.hidden,ledger.total)];
    eprintln!("{}",lines[0]);
    let budget=usize::try_from(budget).map_err(|_|"budget exceeds the address space")?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,budget,|backend| {
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,model,&mut store,workers)?;backend.synchronize()?;
        let files=load::Files::open(&checkpoint,model)?;
        let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
        lines.push(format!("LOOP-ROWS-PRICE LOAD tensors={} stacks={} fingerprint={} device_bytes={}",loaded.report.tensors,loaded.operands.stacks().len(),loaded.fingerprint(),backend.allocated_bytes()));
        eprintln!("{}",lines[1]);
        let device=Device::new(backend);
        let counters=||{let s=backend.stats();[s.queued_calls,s.stream_wait_calls,s.stream_wait_ns,s.dtoh_calls,s.host_launch_ns,s.output_download_ns,s.allocation_ns]};
        let (mut part,equal)=price_rows(&device,&weights,d,&ids,&tops,&prefixes,&rows,&profile,repeats,&||backend.synchronize(),&counters)?;
        lines.append(&mut part);
        receipt(out,&lines,equal)
    })
}
/// Mode `loop-read-ceiling`: the rate of a plain device-to-device copy on this device, a ceiling to hold a product
/// that reads its weights once against. Two vectors of 2^28 words (1 GiB each): one copy to warm, then 8 copies
/// on the stream with one wait at the end. A copy reads 1 GiB and writes 1 GiB; no word is compared.
#[cfg(target_os="linux")]
pub fn run_ceiling(out:&Path)->Result<()> {
    const WORDS:usize=1<<28;const COPIES:u32=8;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,8<<30,|backend| {
        let source=backend.vector(WORDS)?;let mut target=backend.vector(WORDS)?;
        backend.copy_range(&source,&mut target,0,0,WORDS)?;backend.synchronize()?;
        let started=Instant::now();
        for _ in 0..COPIES {backend.copy_range(&source,&mut target,0,0,WORDS)?;}
        backend.synchronize()?;
        let (seconds,bytes)=(started.elapsed().as_secs_f64(),(WORDS*4) as f64*f64::from(COPIES));
        receipt(out,&[format!("LOOP-READ-CEILING copies={COPIES} bytes_read_a_copy={} seconds={seconds:.6} read_gb_a_second={:.1} read_and_write_gb_a_second={:.1}",
            WORDS*4,bytes/seconds/1.0e9,2.0*bytes/seconds/1.0e9)],true)
    })
}
/// Mode `loop-token-time`: the real checkpoint on one device.
#[cfg(target_os="linux")]
pub fn run(model:&Path,trace:&Path,out:&Path,new:&str)->Result<()> {run_with(model,trace,out,new,None,false)}
/// Mode `loop-token-time-pooled`: mode `loop-token-time` with the vector pool of the backend on, as a server sets it.
#[cfg(target_os="linux")]
pub fn run_pooled(model:&Path,trace:&Path,out:&Path,new:&str)->Result<()> {run_with(model,trace,out,new,None,true)}
/// The device run of both modes. `rows` is the step row limit and the count of prompt rows of mode `loop-rows-time`.
/// `pool` turns the vector pool of the backend on (margin 1 GiB) after the load.
#[cfg(target_os="linux")]
fn run_with(model:&Path,trace:&Path,out:&Path,new:&str,rows:Option<(Limits,usize)>,pool:bool)->Result<()> {
    use engine_model::plan::{loop_load,loop_state::{LoopLedger,LoopStatePlan,context_limit}};
    use spark_tp::{bounded_decoder::Device,loop_decoder::operands::{device::DeviceStore,load,weights::LoadedWeights}};
    let started=Instant::now();
    let new=new.parse::<usize>().map_err(|_|format!("loop token time: {new} is no count of new tokens"))?;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut prompt,trace_tops)=trace_ids(trace)?;
    if let Some((_,positions))=rows {
        if positions>prompt.len() || positions>trace_tops.len() {return Err(format!("loop rows time: the folder has {} prompt IDs and {} top rows; POSITIONS is {positions}",prompt.len(),trace_tops.len()));}
        prompt.truncate(positions);
    }
    let checkpoint=loop_load::read_folder(model).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    let (d,capacity)=(&checkpoint.description,(prompt.len()+new) as u64);
    // The memory rule of the load check: the free bytes less the larger of one tenth and 2 GiB, and the ledger of
    // the weights and of one chat of this length must fit.
    let bucket=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
    let state=LoopStatePlan::new(d,capacity,bucket).map_err(|e|e.to_string())?;
    let ledger=LoopLedger::new(&checkpoint.load.bytes,&state).map_err(|e|e.to_string())?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let budget=free.saturating_sub((free/10).max(2<<30)) as u64;
    let limit=context_limit(d,&checkpoint.load.bytes,bucket,budget).map_err(|e|e.to_string())?;
    ledger.admit(budget,limit).map_err(|refusal|format!("text service memory refusal: {refusal}"))?;
    let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
    let mut lines=vec![format!("LOOP-TOKEN-TIME PLAN backend=device prompt_tokens={} new_tokens={new} visits={} width={} head_rows={} capacity={capacity} ledger_bytes={} budget_bytes={budget}",prompt.len(),d.visits.len(),d.hidden,d.head_rows,ledger.total)];
    eprintln!("{}",lines[0]);
    let budget=usize::try_from(budget).map_err(|_|"budget exceeds the address space")?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,budget,|backend| {
        let image=started.elapsed().as_secs_f64();
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,model,&mut store,workers)?;backend.synchronize()?;
        let files=load::Files::open(&checkpoint,model)?;
        let weights=LoadedWeights::new(&checkpoint,&files,&loaded.operands)?;
        if pool {backend.configure_vector_pool(1<<30);eprintln!("LOOP-TOKEN-TIME POOL margin_bytes={}",1usize<<30);}
        lines.push(format!("LOOP-TOKEN-TIME LOAD tensors={} units={} fingerprint={} host_weight_bytes={} device_bytes={} seconds_to_backend={image:.3} seconds_to_weights={:.3}",
            loaded.report.tensors,loaded.report.units,loaded.fingerprint(),weights.host_bytes(),backend.allocated_bytes(),started.elapsed().as_secs_f64()));
        eprintln!("{}",lines[1]);
        if pool {backend.configure_vector_pool(1<<30);lines.push("LOOP-TOKEN-TIME POOL vector_pool=on margin_bytes=1073741824".into());}
        let device=Device::new(backend);
        if let Some((limits,positions))=rows {
            let (mut part,equal)=profile_rows(&device,&weights,d,&prompt,new,limits,&||backend.synchronize(),&||[backend.allocated_bytes(),backend.peak_accounted_bytes()],Some(trace_tops[positions-1]))?;
            lines.append(&mut part);
            return receipt(out,&lines,equal);
        }
        let counters=||{let s=backend.stats();[s.queued_calls,s.stream_wait_calls,s.stream_wait_ns,s.dtoh_calls,s.host_launch_ns,s.output_download_ns,s.allocation_ns]};
        let (mut part,equal)=profile(&device,&weights,d,&prompt,new,&||backend.synchronize(),&counters,&trace_tops)?;
        lines.append(&mut part);
        // The smallest weight unit that stays on the device and is read through a product.
        let smallest=weights.names().filter(|(_,host)|!host).filter_map(|(weight,_)|weights.product_shape(weight).ok().flatten().map(|(rows,columns)|(weight,rows,columns))).min_by_key(|unit|unit.1*unit.2);
        lines.append(&mut floors(&device,&weights,smallest,&||backend.synchronize())?);
        receipt(out,&lines,equal)
    })
}
