//! The block mix, and for a visit inside the loop the messages and the gates.
//!
//! Inside a block the stream holds only what the layers of that block added. A layer reads the past through a
//! softmax blend over the finished block vectors and the stream: the block mix. A visit of a later pass also reads
//! four messages from the pass before, each through its own gate.
//!
//! This file is written over the trait `Compute`, so its host run is the byte reference of its device run.
//! It follows these lines of the reference (`transformers` v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`)
//! and holds its own code: the plain norm, 67-76; the gate, 79-96; the messages, 145-173; the block mix, 828-853;
//! the place of the mixes and the gates in a visit, 876-910, 937-997, 1022-1062 and 1091-1149; what each pass
//! starts from, 1359-1440; the end mix, 1442.
//!
//! Number types: a mix result, a message and a gate result are 32-bit and are not rounded (reference lines 90-96,
//! 161-172 and 839-852).
//! The two norms inside a gate and the inner product of a gate are rounded to BF16.
//!
//! Launches of one mix: the plain norm of the stream (arm `PlainNorm`), one join into the stack, the dots
//! (arm `RowDots`), the softmax and the blend (arm `MixBlend`). The norm of a finished block is made once: at
//! the first layer of a block the norm of the stream is the row of the block before it.
use super::rows::{gather,gather_table,rows_operation,swap,swap_table};
use crate::{Result,bounded_decoder::Compute};
use engine_model::plan::description::{LoopLayerDescription,LoopTextDescription,Visit};
use spark_gpu_probe::{causal_ops::{Op,Shape},prompt_ops};
#[cfg(test)] pub(super) mod support;
#[cfg(test)] mod tests;

/// Epsilon of the plain norm inside a mix and a message. The reference builds that norm with its own default and
/// not with the norm epsilon of the config (reference lines 68, 152 and 833). The norms of a gate take the config's.
pub const MIX_NORM_EPSILON:f32=1.0e-6;
/// Messages that a loop layer reads, in the order of `Message` (reference line 149).
pub const MESSAGES:usize=4;
/// Query rows of an attention mix with separate inputs: query, key, value (reference lines 1037-1039).
pub const ATTENTION_INPUTS:usize=3;

/// The message that a gate mixes in.
#[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub enum Message {Query,Key,Value,Expert}
/// A weight of the driver, by what it is. This is the one list of both halves of a visit and of the token step.
/// A number is the stored layer; a second number is the expert. A shape is `[rows, columns]`, row-major.
/// A matrix "for `product`" is read through `Weights::product` only. Every other weight is read as host words.
/// Taps "by lag" are as arm `ShortConv` reads them: channel (or head), then lag, and lag 0 meets the current token.
#[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub enum Weight {
    // The token step.
    /// The stored embedding table: `[stored rows, width]`, BF16 values. The host reads one row of it.
    Embedding,
    /// The weight of the norm of the embedding: `[width]`.
    EmbeddingNorm,
    /// The weight of the last norm: `[width]`.
    LastNorm,
    /// The output head, only the rows that the head can write: `[head rows, width]`, for `product`.
    Head,
    // The attention half.
    /// The weights of the norms of the query, key and value inputs: `[width]` each. A layer with one attention norm
    /// answers all three with it.
    QueryNorm(usize),
    KeyNorm(usize),
    ValueNorm(usize),
    /// `[query heads * head size, width]`, for `product`.
    Query(usize),
    /// `[key heads * head size, width]`, for `product`.
    Key(usize),
    /// `[key heads * head size, width]`, for `product`.
    Value(usize),
    /// The relative position projection: `[query heads * relative width, width]`, for `product`.
    Relative(usize),
    /// `[query heads, width]`, for `product`.
    Sink(usize),
    /// `[query heads * head size, width]`, for `product`.
    OutputGate(usize),
    /// `[width, query heads * head size]`, for `product`.
    Output(usize),
    /// The learned scale of the query clip, as arm `HeadClip` reads it: `[query heads, head size]`.
    QueryClip(usize),
    /// The learned scale of the clip of the relative state, as arm `HeadClip` reads it: `[query heads, relative width]`.
    RelativeClip(usize),
    /// The distance table, as arm `RelBias` reads it: `[relative width, extent]`, row then distance.
    RelativeTable(usize),
    /// The fixed taps of the key convolution, by lag: `[key heads * head size, taps]`.
    KeyTaps(usize),
    /// The fixed taps of the value convolution, by lag: `[key heads * head size, taps]`.
    ValueTaps(usize),
    /// The fixed taps of the convolution after attention, by lag: `[width, taps]`.
    ConvAfterAttention(usize),
    /// The projection that makes the key taps of a token, in a layer that has it: `[key heads * taps, width]`, for
    /// `product`. Row = head * taps + lag.
    KeyTapProjection(usize),
    /// The same for the value taps.
    ValueTapProjection(usize),
    // The mixes, the messages and the gates.
    /// Query rows of the attention mix: `[rows, width]`, one row, or three in the order query, key, value. 32-bit.
    MixAttention(usize),
    /// The one query of the expert mix: `[width]`. 32-bit.
    MixExperts(usize),
    /// The one query of the end mix: `[width]`. 32-bit.
    MixOutput,
    /// The message queries of every loop layer: `[loop layers, MESSAGES, width]`, in the order of `Message`. 32-bit.
    Messages,
    /// The inner matrix of a gate: `[gate rank, 2 * width]`, for `product`. Its input is the normed value, then the
    /// normed message.
    GateDown(usize,Message),
    /// `[width, gate rank]`, for `product`.
    GateForget(usize,Message),
    /// `[width, gate rank]`, for `product`.
    GateGain(usize,Message),
    // The expert half.
    /// The weight of the norm before the experts: `[width]`, BF16 values.
    ExpertNorm(usize),
    /// `[experts, width]`, for `product`.
    Router(usize),
    /// The bias of the router pick: `[experts]`. 32-bit.
    RouterBias(usize),
    /// One expert's gate-and-up matrix: `[2 * expert width, width]`, for `product`. The gate rows come first, then
    /// the up rows: a stored matrix that mixes the two kinds of row is put in this order at load.
    GateUp(usize,usize),
    /// One expert's down matrix: `[width, expert width]`, for `product`.
    Down(usize,usize),
    /// The fixed taps of the convolution after the experts, by lag: `[width, taps]`.
    ConvAfterExperts(usize),
}
/// Where a half of a visit gets its weights. A fixture answers from host words. The operands of a loaded checkpoint
/// answer a product from a weight that stays on the device.
pub trait Weights<C:Compute> {
    type RoutePlan;
    /// Select the arithmetic phase for products whose device implementation has separate prompt and decode forms.
    /// Returns the old phase so the driver can restore it after a failed step. Host weights ignore the phase.
    fn arithmetic_phase(&self,_decode:bool)->bool {false}
    /// The 32-bit words of a small weight, in the layout that `Weight` gives.
    fn words(&self,weight:Weight)->Result<&[f32]>;
    /// The matrix times one input row. Each output word is one chain over the columns in rising order from +0.0, a
    /// product and its add are two rounded 32-bit steps, and the sum is not rounded. `chain_product` is the host form.
    fn product(&self,compute:&C,weight:Weight,x:&C::Value)->Result<C::Value>;
    /// The stored rows `tokens` of a table that the owner of the weights holds (the embedding), each value as its 32-bit
    /// word, `[tokens][columns]`, on the compute: the words of the host rows of `words`. None: the owner holds the
    /// table as host words (`words`), and the caller reads them there.
    fn stored_rows(&self,_compute:&C,_weight:Weight,_tokens:&[u32])->Result<Option<C::Value>> {Ok(None)}
    /// `product` of each weight on the same input row, in the order of `weights`: the words of `product` each. The
    /// body here runs one `product` a weight. An owner of weights that holds the matrices one after another gives its
    /// own body: one product over all their rows, cut into the part of each weight.
    fn stack_products(&self,compute:&C,weights:&[Weight],x:&C::Value)->Result<Vec<C::Value>> {weights.iter().map(|&weight|self.product(compute,weight,x)).collect()}
    /// `product` for `rows` input rows. `x` is `[rows][columns]` and the result is `[rows][matrix rows]`, row-major.
    /// Each word is the chain of `product`, and with one row this is `product`. The body here runs one `product` a
    /// row. An owner of weights that has a launch for many rows gives its own body with the same words.
    fn product_rows(&self,compute:&C,weight:Weight,x:&C::Value,rows:usize)->Result<C::Value> {
        need(rows>0 && compute.words(x).is_multiple_of(rows),"a product for rows holds no whole count of input rows")?;
        if rows==1 {return self.product(compute,weight,x);}
        let columns=compute.words(x)/rows;
        let outs=(0..rows).map(|row|self.product(compute,weight,&compute.range(x,row*columns,columns)?)).collect::<Result<Vec<_>>>()?;
        compute.join(&outs.iter().collect::<Vec<_>>())
    }
    /// The products of several matrices of one shape in one call, back to back in the order of `weights`: each the
    /// words of `product` on its matrix. `x` is the input of every matrix (`shared`), or the inputs back to back, one
    /// a matrix. None: no grouped product here; the caller calls `product` one matrix at a time.
    fn products(&self,_compute:&C,_weights:&[Weight],_x:&C::Value,_shared:bool)->Result<Option<C::Value>> {Ok(None)}
    /// `product` of matrix `g` of `weights` (all of one shape) on its own input row `xs[g]`, in one call, back to back
    /// in the order of `weights`: each part the words of `product` on its matrix and input. None: no such call here;
    /// the caller calls `product` one matrix at a time.
    fn group_products(&self,_compute:&C,_weights:&[Weight],_xs:&[&C::Value])->Result<Option<C::Value>> {Ok(None)}
    /// `stack_products` for `rows` input rows (`x` is `[rows][columns]`): where the owner holds the matrices one after
    /// another, one product for many rows over all their rows, so row `r` of the result is the products of all
    /// matrices on input row `r`, side by side (`[rows][matrix rows of all]`). Each word is the chain of `product`.
    /// None: no such stack here; the caller runs `product_rows` a matrix.
    fn stack_product_rows(&self,_compute:&C,_weights:&[Weight],_x:&C::Value,_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// `products` on the experts that the route picked, in pick order, with their numbers left where `ids` is: `ids`
    /// holds the picked numbers as whole 32-bit floats. `first` is expert 0 of the projection and layer
    /// (`Weight::GateUp(layer,0)` or `Weight::Down(layer,0)`). The words are those of `products` on the weights of
    /// those numbers. None: no such call here; the caller reads the numbers and calls `products`.
    fn picked_products(&self,_compute:&C,_first:Weight,_ids:&C::Value,_x:&C::Value,_shared:bool,_token_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Project the gate and up rows and apply the capped activation. The cap words are
    /// gate, up, reciprocal gate and reciprocal up, in that order.
    fn picked_gate(&self,_compute:&C,_layer:usize,_ids:&C::Value,_x:&C::Value,_shared:bool,_caps:[f32;4])->Result<Option<C::Value>> {Ok(None)}
    /// Picked packed products reading strided route rows (`[rows][id_stride]`) with input row repeat (`0` for shared,
    /// `picks` to repeat each token row `picks` times, `1` for one input row per pick) and optional fused `CappedGate`.
    #[allow(clippy::too_many_arguments)]
    fn picked_products_strided(&self,_compute:&C,_first:Weight,_route:&C::Value,_rows:usize,_picks:usize,_id_stride:usize,_x:&C::Value,_repeat:usize,_caps:Option<[f32;4]>)->Result<Option<C::Value>> {Ok(None)}
    /// `picked_products_strided` with an optional compacted LPT dispatch table built by `router_pick_compact`.
    #[allow(clippy::too_many_arguments)]
    fn picked_products_strided_compact(&self,compute:&C,first:Weight,route:&C::Value,rows:usize,picks:usize,id_stride:usize,x:&C::Value,repeat:usize,caps:Option<[f32;4]>,_compact:Option<&C::Value>)->Result<Option<C::Value>> {
        self.picked_products_strided(compute,first,route,rows,picks,id_stride,x,repeat,caps)
    }
    /// Fused `stack_product_rows` writing directly into separate `[rows][part_i]` outputs for each weight in `weights`
    /// with zero intermediate buffer or copy. None: caller falls back to `product_rows`.
    fn stack_products_rows(&self,_compute:&C,_weights:&[Weight],_x:&C::Value,_rows:usize)->Result<Option<Vec<C::Value>>> {Ok(None)}
    /// Dual-input `product_rows` over `[x1[r], x2[r]]` (`[rows][cols/2]` each) without joining or swapping inputs.
    fn product_rows_pair(&self,_compute:&C,_weight:Weight,_x1:&C::Value,_x2:&C::Value,_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Dual-input `product_rows` over `[x1[r], x2[r]]` with BF16 rounding fused into the output store.
    fn product_rows_pair_rounded(&self,_compute:&C,_weight:Weight,_x1:&C::Value,_x2:&C::Value,_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Grouped dual-input `product_rows` over up to 3 narrow matrices of one shape (`GateDown` Q/K/V), each on its own
    /// `(x1[g], x2[g])` pair of `[rows][cols/2]` inputs with BF16 rounding fused into the output store, returning
    /// `[groups][rows][matrix_rows]`.
    fn group_product_rows_pair_rounded(&self,_compute:&C,_weights:&[Weight],_pairs:&[(&C::Value,&C::Value)],_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Device-side deterministic MoE router grouping (`MOE_GROUP_ROUTE`): eliminates per-visit GPU-to-CPU sync for
    /// Prefill routing returns the weights and a route plan, or None to fall back to host grouping.
    fn device_route_group(&self,_compute:&C,_layer:usize,_route:&C::Value,_rows:usize,_picks:usize,_experts:usize)->Result<Option<(C::Value,Self::RoutePlan)>> {Ok(None)}
    /// Device-routed tensor-core MoE expert stack GEMM (`PLAIN_TC_ROWS` Mode 1) using the plan from
    /// `device_route_group`. `first` is `Weight::GateUp(layer, 0)` or `Weight::Down(layer, 0)`. None to fall back.
    fn device_routed_products(&self,_compute:&C,_first:Weight,_route:&Self::RoutePlan,_x:&C::Value)->Result<Option<C::Value>> {Ok(None)}
    /// Prefill: the gate-and-up product of the routed experts of `layer` followed by `CappedGate` in one launch
    /// (`[picks][inner]`, the words of `device_routed_products(GateUp)` then `CappedGate`). `caps` = gate cap, up cap,
    /// reciprocal gate cap, reciprocal up cap. None to fall back to the two launches.
    fn device_routed_gate(&self,_compute:&C,_layer:usize,_route:&Self::RoutePlan,_x:&C::Value,_caps:[f32;4])->Result<Option<C::Value>> {Ok(None)}
    /// Prefill: the down product of the routed experts of `layer` and its `RouteFold` over `weights` (`[rows][picks]`,
    /// pick order) in one call: `[rows][width]`, the words of `device_routed_products(Down)` then `RouteFold`, with
    /// the down words kept as BF16 between the two (the fold rounds each with `rn`). None to fall back to the two.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_down_fold(&self,_compute:&C,_layer:usize,_route:&Self::RoutePlan,_x:&C::Value,_weights:&C::Value,_picks:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Prefill: `device_routed_gate` then `device_routed_down_fold` in one call (`[rows][width]`, their words), the
    /// gated words kept between the two as the BF16 values that the down product reads. None to fall back to the two.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_experts(&self,_compute:&C,_layer:usize,_route:&Self::RoutePlan,_x:&C::Value,_caps:[f32;4],_weights:&C::Value,_picks:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Prefill: `device_routed_experts` reading packed BF16 `x` (`rows * width / 2` words), skipping `TC_MODE_PACK`.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_experts_half_in(&self,_compute:&C,_layer:usize,_route:&Self::RoutePlan,_x:&C::Value,_caps:[f32;4],_weights:&C::Value,_picks:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Prefill: `device_routed_experts` fused with `AfterExperts ShortConv` + `next_ring`, returning `(output, next_ring)`
    /// without materializing the `[rows][width]` intermediate folded expert sum. None to fall back.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_experts_close(
        &self,_compute:&C,_layer:usize,_route:&Self::RoutePlan,_x:&C::Value,_caps:[f32;4],
        _weights:&C::Value,_picks:usize,_residual:&C::Value,_fixed:&[f32],_ring:&C::Value,
        _taps:usize,_first_slot:usize,_bound:f32,
    )->Result<Option<(C::Value,C::Value)>> {Ok(None)}
    /// `device_routed_experts_close` reading packed BF16 `x` (`rows * width / 2` words), skipping `TC_MODE_PACK`.
    #[allow(clippy::too_many_arguments)]
    fn device_routed_experts_close_half_in(
        &self,_compute:&C,_layer:usize,_route:&Self::RoutePlan,_x:&C::Value,_caps:[f32;4],
        _weights:&C::Value,_picks:usize,_residual:&C::Value,_fixed:&[f32],_ring:&C::Value,
        _taps:usize,_first_slot:usize,_bound:f32,
    )->Result<Option<(C::Value,C::Value)>> {Ok(None)}
    /// Prefill fused `GateOuter (GateForget+GateGain) + GateBlend + RoundedNorm` for 1..=3 gates (`TC_MODE_GATE_BLEND_NORM`),
    /// reading FP32 `inner` directly (rounding to BF16 in registers) and eliminating the `[rows, 2 * width]` gate buffer.
    #[allow(clippy::too_many_arguments)]
    fn gate_outer_blend_norm(
        &self,_compute:&C,_layer:usize,
        _entries:&[(Message,&C::Value,&C::Value,&C::Value,Option<&[f32]>)],
        _rows:usize,_eps:f32,_half_out:bool,
    )->Result<Option<Vec<C::Value>>> {Ok(None)}
    /// True when prefill attention projections of `layer` can consume packed BF16 (`HALF_IN`) directly.
    fn prefers_half_attention(&self,_layer:usize)->bool {false}
    /// True when prefill router + packed MoE experts of `layer` can consume packed BF16 (`HALF_IN`) directly.
    fn prefers_half_experts(&self,_layer:usize)->bool {false}
    /// True when loop layers `start..end` can consume packed BF16 gate-normed messages (`TC_MODE_NARROW_HALF`).
    fn prefers_half_messages(&self,_start:usize,_end:usize)->bool {false}
    /// True when `Weight::Output(layer)` can consume packed BF16 attention output (`FW_HALF_OUT`) directly.
    fn prefers_half_output(&self,_layer:usize)->bool {false}
    /// Prefill `FLASH_WINDOW_ATTN` emitting packed BF16 output (`FW_HALF_OUT`).
    fn window_rows_half(&self,_compute:&C,_a:&crate::bounded_decoder::WindowOne<'_,C::Value>,_first_position:u64)->Result<Option<C::Value>> {Ok(None)}
    /// Prefill `MIX_ONE_POST_HALF_OUT`: emits packed BF16 `post` (`rows * asked * dim / 2` words).
    #[allow(clippy::too_many_arguments)]
    fn mix_rows_normed_half(
        &self,_compute:&C,_rows:usize,_dim:usize,_scale:f32,_eps:f32,_post_eps:f32,
        _transpose:bool,_keep_normed:bool,_keep_mixed:bool,
        _queries:&[f32],_list:&[&C::Value],_stream:&C::Value,
        _post_weight:Option<&[f32]>,_post_weight_per_query:bool,
    )->Result<Option<(Option<C::Value>,Option<C::Value>,C::Value)>> {Ok(None)}
    /// Prefill `MIX_LIST_NORM_HALF`: FP32 messages + packed BF16 gate-normed messages.
    #[allow(clippy::too_many_arguments)]
    fn mix_prefill_normed_half(&self,_compute:&C,_rows:usize,_dim:usize,_scale:f32,_eps:f32,_queries:&[f32],_list:&[&C::Value])->Result<Option<(C::Value,C::Value)>> {Ok(None)}
    /// Prefill BF16-input dense product (`HALF_IN`, skips `TC_MODE_PACK`, uses `TC_MODE_NARROW_HALF` for `rows <= 128`).
    fn product_rows_half_in(&self,_compute:&C,_weight:Weight,_x:&C::Value,_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Prefill BF16-input split multi-output product (`HALF_IN`, skips `TC_MODE_PACK`).
    fn stack_products_rows_half_in(&self,_compute:&C,_weights:&[Weight],_x:&C::Value,_rows:usize)->Result<Option<Vec<C::Value>>> {Ok(None)}
    /// Prefill dual-input BF16 product over packed BF16 halves (`TC_MODE_NARROW_HALF` for `rows <= 128`).
    fn product_rows_pair_half_in(&self,_compute:&C,_weight:Weight,_x1:&C::Value,_x2:&C::Value,_rows:usize)->Result<Option<C::Value>> {Ok(None)}
    /// The products of the expert runs of a step of rows, in launch order (the index rows of the rows tile, kind 92).
    /// `runs` holds the matrix and the count of rows of each run, in launch order; together they take `n` rows.
    /// Launch row `i` reads row `input[i]` of `x` (rows of `columns` words), or row `i` of `x` when `input` is None.
    /// Row `j` of the result is launch row `output[j]`, or launch row `j` when `output` is None. Each word is the
    /// chain of `product`: a table changes only where a row is read and where it goes.
    /// The body here gathers the input rows (one launch), runs `product_rows` for each run, joins them, and gathers
    /// the result rows (one launch): the launches of before. An owner of weights with a rows tile that reads and
    /// writes rows through index tables gives its own body with the same words and no gather.
    /// Refused before any launch (`check_runs`).
    fn product_runs(&self,compute:&C,runs:&[(Weight,usize)],x:&C::Value,columns:usize,input:Option<&[usize]>,output:Option<&[usize]>)->Result<C::Value> {
        gathered_runs(self,compute,runs,x,columns,input,output)
    }
}
/// The body of `Weights::product_runs` with gathers: the input rows gathered (one launch), `product_rows` for each run,
/// a join, and the result rows gathered (one launch). An owner of weights with its own body calls it for the cases
/// that its body does not take.
pub fn gathered_runs<C:Compute,W:Weights<C>+?Sized>(weights:&W,compute:&C,runs:&[(Weight,usize)],x:&C::Value,columns:usize,input:Option<&[usize]>,output:Option<&[usize]>)->Result<C::Value> {
    let n=check_runs(compute.words(x),columns,runs,input,output)?;
    let gathered;let x=match input {Some(rows)=>{gathered=super::rows::gather(compute,x,rows,columns)?;&gathered},None=>x};
    let mut at=0;let mut parts=Vec::with_capacity(runs.len());
    for &(weight,count) in runs {parts.push(weights.product_rows(compute,weight,&compute.range(x,at*columns,count*columns)?,count)?);at+=count;}
    let joined=compute.join(&parts.iter().collect::<Vec<_>>())?;
    match output {
        Some(rows)=>{need(compute.words(&joined).is_multiple_of(n),"the products of the runs hold no whole rows")?;super::rows::gather(compute,&joined,rows,compute.words(&joined)/n)},
        None=>Ok(joined),
    }
}
/// The checks of `Weights::product_runs`, on counts only, before any launch. `words` are the words of the input
/// value and `columns` the words of one of its rows. It gives the launch rows of the runs.
/// Refused: a run with no row; an input that holds no whole rows; an input table that does not name one row of the
/// input for each launch row; an output table that is no permutation of the launch rows (each written once).
pub fn check_runs(words:usize,columns:usize,runs:&[(Weight,usize)],input:Option<&[usize]>,output:Option<&[usize]>)->Result<usize> {
    let n=runs.iter().try_fold(0usize,|n,run|if run.1>0 {n.checked_add(run.1)} else {None}).ok_or("loop decoder: a run of products has no rows")?;
    need(n>0 && columns>0 && words.is_multiple_of(columns),"the runs of products have no rows, or their input holds no whole rows")?;
    let source=words/columns;
    match input {
        Some(rows)=>need(rows.len()==n && rows.iter().all(|&row|row<source),"an input row of the runs is outside the input")?,
        None=>need(source==n,"the input of the runs differs from their rows")?,
    }
    if let Some(rows)=output {
        let mut seen=vec![false;n];
        need(rows.len()==n && rows.iter().all(|&row|row<n && !std::mem::replace(&mut seen[row],true)),"the output rows of the runs are no permutation of their rows")?;
    }
    Ok(n)
}
/// The host form of `Weights::product` for a matrix that is held as 32-bit words, row-major.
pub fn chain_product<C:Compute>(compute:&C,matrix:&[f32],x:&C::Value)->Result<C::Value> {
    let cols=compute.words(x);need(cols>0 && !matrix.is_empty() && matrix.len().is_multiple_of(cols),"product matrix extent differs from its input")?;
    let mut s=shape(Op::Project);s.rows=number(matrix.len()/cols)?;s.cols=number(cols)?;
    compute.operation(s,[Some(x),None,None,None],matrix)
}
pub(super) fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(format!("loop decoder: {why}"))}}
pub(super) fn shape(op:Op)->Shape {Shape {op,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}}
pub(super) fn number(n:usize)->Result<u32> {u32::try_from(n).map_err(|_|"loop decoder: a dimension exceeds the operation ABI".to_owned())}
/// The number that stands for a division by `divisor`: the reciprocal, made in 64-bit and rounded to 32-bit once.
/// Where the reference divides a 32-bit tensor by a number, its GPU multiplies by this value: measured on 4,000,000
/// values each of four divisors (the lab receipt `divform.json` of the reference answers). An arm multiplies by it
/// and never divides. Every number that the driver makes from the config is made in 64-bit and rounded once.
pub fn reciprocal(divisor:f64)->f32 {(1.0/divisor) as f32}
/// The score scale of a mix and of a message: the 32-bit value of `width^(-1/2)`, made in 64-bit and rounded once
/// (reference lines 171 and 851: one product with a number, not a division of the tensor).
pub fn mix_scale(width:usize)->f32 {reciprocal((width as f64).sqrt())}

/// The inputs of the attention norms of one visit. `Same`/`Each` are 32-bit and not normed yet;
/// `NormedSame`/`NormedEach` are already weighted-`RoundedNorm`ed (BF16 in FP32);
/// `NormedHalfSame`/`NormedHalfEach` are packed BF16 (`rows * width / 2` words) for zero-pack prefill TC.
#[derive(Debug)]
pub enum Inputs<V> {
    /// One vector is the query, the key and the value input.
    Same(V),
    /// Query, key, value.
    Each([V;ATTENTION_INPUTS]),
    /// Already weighted-normed single input (BF16 values in FP32).
    NormedSame(V),
    /// Already weighted-normed query, key, value inputs (BF16 values in FP32).
    NormedEach([V;ATTENTION_INPUTS]),
    /// Already weighted-normed single input packed as BF16 (`rows * width / 2` words).
    NormedHalfSame(V),
    /// Already weighted-normed query, key, value inputs packed as BF16 (`rows * width / 2` words each).
    NormedHalfEach([V;ATTENTION_INPUTS]),
}
/// Representation of the expert input produced by `Token::expert_input_normed`.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) enum ExpertNorm {Raw,Normed,NormedHalf}
/// What a half of a visit works with: the backend, the weights and the description of the checkpoint.
/// `tables` holds the index table of each swap of a step on the backend, by its two counts: a swap of a step of
/// rows uploads its table once a step, not once a call (`Run::swap`; `clear_tables` at the start of each step).
pub struct Run<'a,C:Compute,W:Weights<C>> {pub compute:&'a C,pub weights:&'a W,pub description:&'a LoopTextDescription,width:usize,scale:f32,
    tables:std::cell::RefCell<std::collections::BTreeMap<(usize,usize),C::Value>>,spreads:std::cell::RefCell<std::collections::BTreeMap<(usize,usize),C::Value>>}
impl<'a,C:Compute,W:Weights<C>> Run<'a,C,W> {
    pub fn new(compute:&'a C,weights:&'a W,description:&'a LoopTextDescription)->Result<Self> {
        let d=description;let width=usize::try_from(d.hidden).map_err(|_|"loop decoder: the width exceeds this host".to_owned())?;
        need(width>0 && !d.layers.is_empty(),"the description has no width or no layer")?;
        need(d.residual.normalize && d.residual.block_layers>0,"a block mix over vectors that are not normed is not built")?;
        need(d.core.start<d.core.end && d.core.end<=d.layers.len() as u64 && d.core.passes>0,"the loop range is outside the layers")?;
        // The inner width of a gate. The repo's path check reads a comparison with a name that ends in "rank" as a
        // choice by machine count, so the value has another name here.
        let gate_width=d.core.gate_rank;
        need(d.core.passes==1 || gate_width>0,"a loop with messages needs a gate rank")?;
        need(d.epsilon>0.0,"the norm epsilon of the description is not above zero")?;
        Ok(Self {compute,weights,description,width,scale:mix_scale(width),tables:Default::default(),spreads:Default::default()})
    }
    /// `rows::swap` with the index table of its two counts uploaded once and kept until `clear_tables`.
    pub(super) fn swap(&self,x:&C::Value,first:usize,second:usize,width:usize)->Result<C::Value> {
        if first==1 || second==1 {return swap(self.compute,x,first,second,width);}
        let mut tables=self.tables.borrow_mut();
        let table=match tables.entry((first,second)) {
            std::collections::btree_map::Entry::Occupied(kept)=>kept.into_mut(),
            std::collections::btree_map::Entry::Vacant(new)=>new.insert(self.compute.upload_constant(&swap_table(first,second))?),
        };
        gather_table(self.compute,x,table,first*second,width)
    }
    /// Each of `rows` rows of `x` (rows of `width` words) `each` times in a row: row `i` of the result is row `i / each`
    /// of `x` (`rows::gather`, the same launch and words). Its index table is uploaded once a step and kept until
    /// `clear_tables`, like the tables of `swap`: an upload ends the launches that the backend has recorded before it
    /// (measured: one upload of 128 bytes a visit, 48 graph launches a 4-row verify pass).
    pub(super) fn spread(&self,x:&C::Value,rows:usize,each:usize,width:usize)->Result<C::Value> {
        need(rows>0 && each>0 && self.compute.words(x)==rows*width,"a spread differs from its rows")?;
        if each==1 {return self.compute.range(x,0,rows*width);}
        let mut spreads=self.spreads.borrow_mut();
        let table=match spreads.entry((rows,each)) {
            std::collections::btree_map::Entry::Occupied(kept)=>kept.into_mut(),
            std::collections::btree_map::Entry::Vacant(new)=>new.insert(self.compute.upload_constant(&(0..rows*each).map(|at|(at/each) as f32).collect::<Vec<_>>())?),
        };
        gather_table(self.compute,x,table,rows*each,width)
    }
    /// Let the index tables of the swaps and spreads go: a step of another count of rows has other tables.
    pub(super) fn clear_tables(&self) {self.tables.borrow_mut().clear();self.spreads.borrow_mut().clear();}
    /// The width of the stream.
    pub fn width(&self)->usize {self.width}
    /// The rows of a value that holds whole rows of the width: the rows of a step.
    pub fn rows(&self,x:&C::Value)->Result<usize> {
        let words=self.compute.words(x);need(words>0 && words.is_multiple_of(self.width),"a value of a step holds no whole count of rows of the width")?;
        Ok(words/self.width)
    }
    /// The stored layer of a visit and its description.
    pub fn layer(&self,visit:&Visit)->Result<(usize,&'a LoopLayerDescription)> {
        let layer=usize::try_from(visit.layer).map_err(|_|"loop decoder: a visit names no layer".to_owned())?;
        Ok((layer,self.description.layers.get(layer).ok_or("loop decoder: a visit names a layer outside the description")?))
    }
    /// The plain norm of `rows` rows, 32-bit and not rounded: arm `PlainNorm`, with the squares on the 32-lane tree.
    fn plain_norm(&self,x:&C::Value,rows:usize)->Result<C::Value> {
        let mut s=shape(Op::PlainNorm);s.heads=number(rows)?;s.dim=number(self.width)?;s.epsilon=MIX_NORM_EPSILON;
        self.compute.operation(s,[Some(x),None,None,None],&[])
    }
    /// A norm of each row of `x` (one or more rows of the width) with a result that is rounded to BF16 once: arm
    /// `RoundedNorm`, with the config's epsilon. `weight` is one row of the width for all rows, or empty for a norm
    /// with no weight. The result is `rn((x * scale) * weight)`, or `rn(x * scale)`. The arm reads the 32-bit input
    /// as it is, and its squares add in the tree of the norm kernel. A row is normed alone: many rows in one launch
    /// give the words of one launch a row.
    /// The driver does not call `Compute::norm`: on a device that call rounds its input first.
    pub(super) fn rounded_norm(&self,x:&C::Value,weight:&[f32])->Result<C::Value> {
        let words=self.compute.words(x);
        need(words>0 && words.is_multiple_of(self.width) && (weight.is_empty() || weight.len()==self.width),"a norm input or its weight differs from the width")?;
        let mut s=shape(Op::RoundedNorm);s.heads=number(words/self.width)?;s.dim=number(self.width)?;s.taps=u32::from(!weight.is_empty());s.epsilon=self.description.epsilon as f32;
        self.compute.operation(s,[Some(x),None,None,None],weight)
    }
    /// One norm inside a gate: no weight. This is the one place where this file norms for a gate.
    fn gate_norm(&self,x:&C::Value)->Result<C::Value> {self.rounded_norm(x,&[])}
    /// The dots of each query row with each of the `count` rows of `stack`: `[queries, count]`, query then stack row.
    /// Each dot is one chain over the width in rising order from +0.0. One call of `Compute::dot_rows` for any count:
    /// arm `RowDots` takes every stack row in one launch, and a device runs the fold entry where it has the shape.
    fn dots(&self,queries:&[f32],stack:&C::Value,count:usize)->Result<C::Value> {self.compute.dot_rows(queries,stack,count,self.width)}
    /// One mix for the `count` rows of a step: every query row blends the plain-normed vectors of `list`. Each entry
    /// of `list` is `[count][width]`: one vector for each row of the step. The result is `[count][queries][width]`, 32-bit.
    /// The scores are the dots times `mix_scale`. Arm `MixBlend` takes the largest score out, adds the exponentials
    /// in list order, divides each by the sum, and adds `weight * vector` in list order from +0.0.
    /// A row of the step reads its own vectors only. With one row the launches are those of a token step.
    fn blend(&self,queries:&[f32],list:&[&C::Value],count:usize)->Result<C::Value> {
        let c=self.compute;let width=self.width;
        need(!queries.is_empty() && queries.len().is_multiple_of(width),"a mix query differs from the width")?;
        need(count>0 && !list.is_empty() && list.iter().all(|row|c.words(row)==count*width),"a mix row differs from the width")?;
        let (asked,kept)=(queries.len()/width,list.len());
        if let Some(mixed)=c.mix_prefill(count,width,self.scale,false,queries,list)? {return Ok(mixed);}
        let joined=c.join(list)?;
        if count>1 && let Some(mixed)=self.blend_listed(queries,&joined,count,kept)? {return Ok(mixed);}
        let (stack,dots)=if count==1 {let dots=self.dots(queries,&joined,kept)?;(joined,dots)} else {
            // `[kept][count][width]` to `[count][kept][width]`: each row of the step with its own vectors.
            let stack=self.swap(&joined,kept,count,width)?;
            let dots=self.row_dots(queries,&stack,count,kept,prompt_ops::MAX_ROWS)?;(stack,dots)
        };
        let mut s=shape(Op::MixBlend);s.heads=number(asked)?;s.cols=number(kept)?;s.dim=number(width)?;s.bound=self.scale;
        rows_operation(c,s,count,[Some(&dots),Some(&stack),None,None],0,&[])
    }
    /// `blend` of a step of `count` rows on the list form of the rows entry (`block_rows::LIST`): `joined` is the block
    /// list as the step made it, `[kept][count][width]`, and arms `RowDots` and `MixBlend` read vector `e` of row `r`
    /// at `(e * count + r) * width`. No swap of the list and none of the dots. Each dot is the chain of `dots` over the
    /// same two vectors, and the blend adds in the same list order, so every word is that of the swapped form.
    /// None: the backend has no rows entry or the counts do not fit it; the caller swaps.
    fn blend_listed(&self,queries:&[f32],joined:&C::Value,count:usize,kept:usize)->Result<Option<C::Value>> {
        let (c,width,list)=(self.compute,self.width,spark_gpu_probe::block_rows::LIST);let asked=queries.len()/width;
        let dots=crate::bounded_decoder::row_dots_shape(queries.len(),kept,width)?;
        let mut s=shape(Op::MixBlend);s.heads=number(asked)?;s.cols=number(kept)?;s.dim=number(width)?;s.bound=self.scale;
        let fits=|s:Shape|spark_gpu_probe::block_rows::lengths(s,count,list).is_some();
        if !fits(dots) || !fits(s) {return Ok(None);}
        let Some(scores)=c.block_rows(dots,count,list,[Some(joined),None,None,None],queries)? else {return Ok(None);};
        c.block_rows(s,count,list,[Some(&scores),Some(joined),None,None],&[])?.map(Some).ok_or_else(||"loop decoder: the rows entry took the dots of a mix but not its blend".to_owned())
    }
    /// `blend` returning directly in transposed `[queries][count][width]` layout: avoids `PermuteRows` swap
    /// when `Compute::mix_blend_transposed` is supported, otherwise falls back to `blend` + `swap`.
    fn blend_transposed(&self,queries:&[f32],list:&[&C::Value],count:usize)->Result<C::Value> {
        let c=self.compute;let width=self.width;
        need(!queries.is_empty() && queries.len().is_multiple_of(width),"a mix query differs from the width")?;
        need(count>0 && !list.is_empty() && list.iter().all(|row|c.words(row)==count*width),"a mix row differs from the width")?;
        let (asked,kept)=(queries.len()/width,list.len());
        if count==1 {return self.blend(queries,list,1);}
        if let Some(mixed)=c.mix_prefill(count,width,self.scale,true,queries,list)? {return Ok(mixed);}
        let list_flag=spark_gpu_probe::block_rows::LIST;
        let dots=crate::bounded_decoder::row_dots_shape(queries.len(),kept,width)?;
        if spark_gpu_probe::block_rows::lengths(dots,count,list_flag).is_some() {
            let joined=c.join(list)?;
            if let Some(scores)=c.block_rows(dots,count,list_flag,[Some(&joined),None,None,None],queries)?
                && let Some(transposed)=c.mix_blend_transposed(count,asked,kept,width,self.scale,&scores,&joined)? {
                return Ok(transposed);
            }
        }
        let mixed=self.blend(queries,list,count)?;
        self.swap(&mixed,count,asked,width)
    }
    /// The dots of a step of `count` rows. Each row of the step has `kept` vectors in `stack`, `[count][kept][width]`,
    /// and every query row meets each of them. The result is `[count][asked][kept]`, the order that arm `MixBlend`
    /// reads for one row. Each dot is the chain of `dots`.
    /// One launch (`prompt_ops::Op::ProjectRows`) takes `most` vectors: the queries on `n * kept` vectors give
    /// `[asked][n][kept]`. Where the step has more vectors, its rows go in parts of whole rows. A part changes no
    /// word: a dot reads one vector.
    fn row_dots(&self,queries:&[f32],stack:&C::Value,count:usize,kept:usize,most:usize)->Result<C::Value> {
        let c=self.compute;let (width,asked)=(self.width,queries.len()/self.width);
        need(kept>0 && kept<=most && c.words(stack)==count*kept*width,"a mix has more vectors a row than one launch takes")?;
        let part=|view:&C::Value,n:usize|->Result<C::Value> {
            let mut s=prompt_ops::Shape::new(prompt_ops::Op::ProjectRows,n*kept).ok_or("loop decoder: the dots of a mix exceed the launch format")?;
            s.rows=number(asked)?;s.cols=number(width)?;need(s.lengths().is_some(),"the dots of a mix exceed the launch format")?;
            self.swap(&c.prompt(s,[Some(view),None,None,None,None],queries)?,asked,n,kept)
        };
        let each=most/kept;
        if count<=each {return part(stack,count);}
        let parts=(0..count).step_by(each).map(|start|{let n=each.min(count-start);part(&c.range(stack,start*kept*width,n*kept*width)?,n)}).collect::<Result<Vec<_>>>()?;
        c.join(&parts.iter().collect::<Vec<_>>())
    }
    /// Fused plain-norm of `stream` and mix over `list` followed by that normed stream (`Compute::mix_one`).
    /// Falls back to separate `plain_norm` and `blend`/`blend_transposed` when `mix_one` is unavailable.
    fn norm_blend(
        &self, queries: &[f32], list: &[C::Value], stream: &C::Value,
        count: usize, transpose: bool, keep_normed: bool,
    ) -> Result<(Option<C::Value>, C::Value)> {
        let refs: Vec<&C::Value> = list.iter().collect();
        if let Some((normed, mixed)) = self.compute.mix_rows(count, self.width, self.scale, MIX_NORM_EPSILON, transpose, keep_normed, queries, &refs, stream)? {
            return Ok((normed, mixed));
        }
        let normed = self.plain_norm(stream, count)?;
        let mut all = refs;
        all.push(&normed);
        let mixed = if transpose { self.blend_transposed(queries, &all, count)? } else { self.blend(queries, &all, count)? };
        Ok((keep_normed.then_some(normed), mixed))
    }
    /// `norm_blend` with one query for a visit that gates the mix, and the gate norm of the mix: (the normed row when
    /// `keep_normed`, the mix, its gate norm). A step of more than `SMALL_PASS_ROWS` rows makes the mix and its gate norm
    /// in one launch where the backend has it (`Compute::mix_prefill_normed`, the words of `blend` and of `gate_norm`);
    /// the gate norm is None otherwise, and the caller runs `gate_norm`. Up to `SMALL_PASS_ROWS` rows (where
    /// `Compute::mix_rows` may take the step) this is `norm_blend` and its launches.
    fn norm_blend_gate(&self,queries:&[f32],stack:&[C::Value],x:&C::Value,count:usize,keep_normed:bool)->Result<(Option<C::Value>,C::Value,Option<C::Value>)> {
        if count<=super::SMALL_PASS_ROWS || queries.len()!=self.width {
            let (normed,mixed)=self.norm_blend(queries,stack,x,count,false,keep_normed)?;
            return Ok((normed,mixed,None));
        }
        let normed=self.plain_norm(x,count)?;
        let mut list=stack.iter().collect::<Vec<_>>();list.push(&normed);
        // One query: the transposed layout `[1][count][width]` is the blend's own `[count][1][width]`.
        let (mixed,gated)=match self.compute.mix_prefill_normed(count,self.width,self.scale,self.description.epsilon as f32,queries,&list)? {
            Some((mixed,gated))=>(mixed,Some(gated)),
            None=>(self.blend(queries,&list,count)?,None),
        };
        drop(list);
        Ok((keep_normed.then_some(normed),mixed,gated))
    }
    /// One gate for each row of a step: `(1 - tanh(F)) * value + tanh(G) * message`, 32-bit. Every value here is
    /// `[rows][width]`. `normed` is `gate_norm(value)` and `message` is the message with its gate norm,
    /// `(message, gate_norm(message))`: the norms of all messages of a pass are one launch (`Token::enter`).
    /// The inner vector is the down matrix times the two rounded norms side by side, rounded to BF16. `F` and `G` are
    /// the forget and the gain matrix times the inner vector, 32-bit. Arm `GateBlend` makes the two products and the sum.
    fn gate(&self,layer:usize,which:Message,value:&C::Value,normed:&C::Value,(message,message_normed):(&C::Value,&C::Value))->Result<C::Value> {
        let c=self.compute;let w=self.weights;let count=self.rows(value)?;let gate_width=self.description.core.gate_rank as usize;
        need([normed,message,message_normed].iter().all(|v|c.words(v)==count*self.width),"a gate input differs from the rows of the step")?;
        if let Some(rounded)=w.product_rows_pair_rounded(c,Weight::GateDown(layer,which),normed,message_normed,count)? {
            need(c.words(&rounded)==count*gate_width,"a gate's inner product differs from the gate rank")?;
            return self.gate_outer(layer,which,value,message,&rounded);
        }
        let inner=match w.product_rows_pair(c,Weight::GateDown(layer,which),normed,message_normed,count)? {
            Some(v)=>v,
            None=>w.product_rows(c,Weight::GateDown(layer,which),&self.gate_pair(count,normed,(message,message_normed))?,count)?,
        };
        need(c.words(&inner)==count*gate_width,"a gate's inner product differs from the gate rank")?;
        self.gate_outer(layer,which,value,message,&c.round(&inner)?)
    }
    /// The down matrix's input of a gate: `normed` and the gate norm of the message side by side for each row.
    fn gate_pair(&self,count:usize,normed:&C::Value,(message,message_normed):(&C::Value,&C::Value))->Result<C::Value> {
        let c=self.compute;let width=self.width;
        need([normed,message,message_normed].iter().all(|v|c.words(v)==count*width),"a gate input differs from the rows of the step")?;
        let joined=c.join(&[normed,message_normed])?;
        if count==1 {Ok(joined)} else {self.swap(&joined,2,count,width)}
    }
    /// The three gates of a one-row visit (query, key, value), gate i on `values[i]`, its norm `normeds[i]` and
    /// `messages[i]`. Their three down products are one call where the owner of the weights has one
    /// (`Weights::group_products`), and the three inner vectors one rounding: each word is the word of its own gate (a
    /// product reads its own matrix and input, the rounding reads one word). None where no such call is here; the
    /// caller runs each gate alone.
    fn gates_one_row(&self,layer:usize,values:[&C::Value;3],normeds:[&C::Value;3],messages:&[(C::Value,C::Value)])->Result<Option<Vec<C::Value>>> {
        let c=self.compute;let gate_width=self.description.core.gate_rank as usize;
        let order=[Message::Query,Message::Key,Message::Value];
        need(messages.len()==order.len(),"the grouped gates take three messages")?;
        for value in values {need(self.rows(value)?==1,"the grouped gates take one row")?;}
        let pairs=normeds.iter().zip(messages).map(|(normed,(m,n))|self.gate_pair(1,normed,(m,n))).collect::<Result<Vec<_>>>()?;
        let downs=order.map(|which|Weight::GateDown(layer,which));
        let Some(inner)=self.weights.group_products(c,&downs,&pairs.iter().collect::<Vec<_>>())? else {return Ok(None)};
        need(c.words(&inner)==order.len()*gate_width,"the grouped inner products differ from the gate rank")?;
        let inner=c.round(&inner)?;
        order.into_iter().zip(values).zip(messages).enumerate().map(|(i,((which,value),(m,_)))|self.gate_outer(layer,which,value,m,&c.range(&inner,i*gate_width,gate_width)?)).collect::<Result<Vec<_>>>().map(Some)
    }
    /// The three gates of a multi-row visit (`count >= 2`, query, key, value): their three dual-input down products
    /// and BF16 rounding run in one grouped 48-warp narrow launch (`Weights::group_product_rows_pair_rounded`).
    fn gates_rows(&self,layer:usize,values:[&C::Value;3],normeds:[&C::Value;3],messages:&[(C::Value,C::Value)],count:usize)->Result<Option<Vec<C::Value>>> {
        let c=self.compute;let gate_width=self.description.core.gate_rank as usize;
        let order=[Message::Query,Message::Key,Message::Value];
        need(messages.len()==order.len(),"the grouped gates take three messages")?;
        let downs=order.map(|which|Weight::GateDown(layer,which));
        let pairs:Vec<(&C::Value,&C::Value)>=normeds.iter().zip(messages).map(|(&normed,(_,n))|(normed,n)).collect();
        let Some(inner)=self.weights.group_product_rows_pair_rounded(c,&downs,&pairs,count)? else {return Ok(None)};
        need(c.words(&inner)==order.len()*count*gate_width,"the grouped inner products differ from the gate rank")?;
        let each=count*gate_width;
        order.into_iter().zip(values).zip(messages).enumerate().map(|(i,((which,value),(m,_)))|self.gate_outer(layer,which,value,m,&c.range(&inner,i*each,each)?)).collect::<Result<Vec<_>>>().map(Some)
    }
    /// Compute the `[count][2 * width]` forget+gain pairs for all three attention gates (`count >= 2`),
    /// running their three dual-input `GateDown` products + BF16 rounding in one grouped narrow launch when supported.
    fn gates_rows_pairs(&self,layer:usize,normeds:[&C::Value;3],messages:&[(C::Value,C::Value)],count:usize)->Result<Vec<C::Value>> {
        let c=self.compute;let gate_width=self.description.core.gate_rank as usize;
        let order=[Message::Query,Message::Key,Message::Value];
        need(messages.len()==order.len(),"the grouped gates take three messages")?;
        let downs=order.map(|which|Weight::GateDown(layer,which));
        let pairs:Vec<(&C::Value,&C::Value)>=normeds.iter().zip(messages).map(|(&normed,(_,n))|(normed,n)).collect();
        if let Some(inner)=self.weights.group_product_rows_pair_rounded(c,&downs,&pairs,count)? {
            need(c.words(&inner)==order.len()*count*gate_width,"the grouped inner products differ from the gate rank")?;
            let each=count*gate_width;
            return order.into_iter().enumerate().map(|(i,which)|self.gate_outer_pair(layer,which,count,&c.range(&inner,i*each,each)?)).collect();
        }
        order.into_iter().zip(normeds).zip(messages).map(|((which,normed),(m,n))|self.gate_pair_outer(layer,which,count,normed,(m,n))).collect()
    }
    /// The forget and gain products of a gate on its rounded inner vector and arm GateBlend (see `gate`).
    fn gate_outer(&self,layer:usize,which:Message,value:&C::Value,message:&C::Value,inner:&C::Value)->Result<C::Value> {
        let pair=self.gate_outer_pair(layer,which,self.rows(value)?,inner)?;
        self.gate_blend_only(&pair,value,message)
    }
    fn gate_blend_only(&self,pair:&C::Value,value:&C::Value,message:&C::Value)->Result<C::Value> {
        let count=self.rows(value)?;
        let mut s=shape(Op::GateBlend);s.dim=number(self.width)?;
        rows_operation(self.compute,s,count,[Some(pair),Some(value),Some(message),None],0,&[])
    }
    /// Compute the `[count][2 * width]` forget+gain gate projections for a multi-row gate (before `GateBlend`).
    fn gate_outer_pair(&self,layer:usize,which:Message,count:usize,inner:&C::Value)->Result<C::Value> {
        let c=self.compute;let w=self.weights;let width=self.width;
        let pair=|first:&C::Value,second:&C::Value|->Result<C::Value> {let joined=c.join(&[first,second])?;if count==1 {Ok(joined)} else {self.swap(&joined,2,count,width)}};
        // One row: the forget and the gain matrix on the same inner vector are one products call; where the owner holds
        // them one after another the two parts lie side by side and the pair below is a view.
        // More rows: where the owner holds them one after another, one product for all rows gives each row's forget and
        // gain side by side, the pair that arm GateBlend reads, with no join and no swap.
        let both=[Weight::GateForget(layer,which),Weight::GateGain(layer,which)];
        let stacked=w.stack_product_rows(c,&both,inner,count)?;
        match stacked {
            Some(pair)=>{need(c.words(&pair)==count*2*width,"a gate's outer product differs from the width")?;Ok(pair)},
            None=>{
                let (forget,gain)=if count==1 {
                    let mut y=w.stack_products(c,&both,inner)?.into_iter();
                    (y.next().ok_or("loop decoder: a forget product is missing")?,y.next().ok_or("loop decoder: a gain product is missing")?)
                } else {(w.product_rows(c,both[0],inner,count)?,w.product_rows(c,both[1],inner,count)?)};
                need(c.words(&forget)==count*width && c.words(&gain)==count*width,"a gate's outer product differs from the width")?;
                pair(&forget,&gain)
            },
        }
    }
    /// Compute the `[count][2 * width]` forget+gain pair from `(normed, (message, message_normed))`.
    fn gate_pair_outer(&self,layer:usize,which:Message,count:usize,normed:&C::Value,(message,message_normed):(&C::Value,&C::Value))->Result<C::Value> {
        let c=self.compute;let w=self.weights;let gate_width=self.description.core.gate_rank as usize;
        need([normed,message,message_normed].iter().all(|v|c.words(v)==count*self.width),"a gate input differs from the rows of the step")?;
        if let Some(rounded)=w.product_rows_pair_rounded(c,Weight::GateDown(layer,which),normed,message_normed,count)? {
            need(c.words(&rounded)==count*gate_width,"a gate's inner product differs from the gate rank")?;
            return self.gate_outer_pair(layer,which,count,&rounded);
        }
        let inner=match w.product_rows_pair(c,Weight::GateDown(layer,which),normed,message_normed,count)? {
            Some(v)=>v,
            None=>w.product_rows(c,Weight::GateDown(layer,which),&self.gate_pair(count,normed,(message,message_normed))?,count)?,
        };
        need(c.words(&inner)==count*gate_width,"a gate's inner product differs from the gate rank")?;
        self.gate_outer_pair(layer,which,count,&c.round(&inner)?)
    }
}

/// The mix state of the rows of one step: it lives for one step and holds no chat state. A token step is a step of
/// one row. Every value that goes in or comes out is `[rows][width]`, row-major, and a row reads its own words only.
/// The stack holds the plain norm of each vector of the block list, oldest first: the embedding, then each finished
/// block. Each entry is `[rows][width]`. `messages` holds the messages of the running pass and their gate norms, each
/// `[messages][rows][width]`, so one message of all rows is one range.
pub struct Token<V> {stack:Vec<V>,normed:Option<V>,start:Option<(V,usize)>,messages:Option<(V,V,bool)>}
impl<V> Default for Token<V> {fn default()->Self {Self {stack:Vec::new(),normed:None,start:None,messages:None}}}
impl<V> Token<V> {
    pub fn new()->Self {Self::default()}
    /// Vectors on the block list now.
    pub fn blocks(&self)->usize {self.stack.len()}
    /// True when the running pass reads messages.
    pub fn has_messages(&self)->bool {self.messages.is_some()}
    /// Call before each visit, with the stream after the visit before it. It gives the stream that the visit reads.
    /// The first loop layer of the first pass saves the restart point: its stream and the block list.
    /// The first loop layer of a later pass makes every message of that pass in one launch, from the block outputs of
    /// the pass before (the last one is `stream`), and restarts from the saved point.
    pub fn enter<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:V)->Result<V> {
        let c=run.compute;let d=run.description;let width=run.width;let count=run.rows(&stream)?;
        if visit.layer!=d.core.start {
            need(visit.messages==self.messages.is_some() || visit.layer>=d.core.end,"a visit and its pass differ on the messages")?;
            // The layers after the loop go on from the last pass and read no message.
            if !visit.messages {self.messages=None;}
            return Ok(stream);
        }
        if !visit.messages {self.start=Some((c.range(&stream,0,count*width)?,self.stack.len()));self.messages=None;return Ok(stream);}
        let (saved,kept)=self.start.as_ref().ok_or("loop decoder: a later pass has no saved start")?;
        // The first loop layer starts a block, so it put the loop input on the list. That row is no output of the pass.
        need(self.stack.len()>*kept && c.words(saved)==count*width,"the pass before left no block list entry, or another count of rows")?;
        let queries=run.weights.words(Weight::Messages)?;
        need(queries.len()==(d.core.end-d.core.start) as usize*MESSAGES*width,"the message queries differ from the loop range")?;
        // No visit reads the messages of the pass before again: they go before the new ones are made.
        self.messages=None;
        // A prompt step of more than `SMALL_PASS_ROWS` rows makes its messages and their gate norms in one launch where
        // the backend has it: the words of the transposed mix and of `gate_norm` below (the norm's epsilon is the
        // config's, as in `rounded_norm`). Up to `SMALL_PASS_ROWS` rows (where `Compute::mix_rows` may take the step)
        // the messages take `norm_blend`.
        let both=if count>super::SMALL_PASS_ROWS {
            let last=run.plain_norm(&stream,count)?;
            let mut from=self.stack[*kept+1..].iter().collect::<Vec<_>>();from.push(&last);
            let want_half=run.weights.prefers_half_messages(d.core.start as usize,d.core.end as usize)
                && d.layers[d.core.start as usize..d.core.end as usize].iter().all(|l|l.attention_queries>0);
            if want_half && let Some((m,n))=run.weights.mix_prefill_normed_half(c,count,width,run.scale,d.epsilon as f32,queries,&from)? {
                Some((m,n,true))
            } else {
                match c.mix_prefill_normed(count,width,run.scale,d.epsilon as f32,queries,&from)? {
                    Some((m,n))=>Some((m,n,false)),
                    None=>{
                        let messages=run.blend_transposed(queries,&from,count)?;
                        // The gate norm of every message of the pass and every row in one launch: each row is normed alone.
                        let normed=run.gate_norm(&messages)?;
                        Some((messages,normed,false))
                    },
                }
            }
        } else {None};
        let (messages,normed,normed_half)=match both {
            Some(triple)=>triple,
            None=>{
                let refs:Vec<&V>=self.stack[*kept+1..].iter().collect();
                if let Some((_,Some(messages),normed))=c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,d.epsilon as f32,true,false,true,queries,&refs,&stream,None,false)? {
                    (messages,normed,false)
                } else {
                    // The mix gives `[messages][rows][width]` directly when fused or transposed blend is available.
                    let (_,messages)=run.norm_blend(queries,&self.stack[*kept+1..],&stream,count,true,false)?;
                    // The gate norm of every message of the pass and every row in one launch: each row is normed alone.
                    let normed=run.gate_norm(&messages)?;
                    (messages,normed,false)
                }
            },
        };
        let restart=c.range(saved,0,count*width)?;
        self.stack.truncate(*kept);self.normed=None;self.messages=Some((messages,normed,normed_half));
        Ok(restart)
    }
    /// One message of a layer for the `count` rows of the step and its gate norm (`[count][width]`, or `[count][width/2]` if half).
    fn message_form<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,layer:usize,spec:&LoopLayerDescription,which:Message,count:usize)->Result<(V,V,bool)> {
        let start=run.description.core.start as usize;
        need(spec.message_gates && layer>=start,"a visit with messages names a layer with no gates")?;
        let (all,normed,half)=self.messages.as_ref().ok_or("loop decoder: a visit reads messages that no pass made")?;
        let idx=(layer-start)*MESSAGES+which as usize;
        let len=count*run.width;
        let n_len=if *half {count*(run.width/2)} else {len};
        Ok((run.compute.range(all,idx*len,len)?,run.compute.range(normed,idx*n_len,n_len)?,*half))
    }
    /// One message of a layer for the `count` rows of the step and its FP32 gate norm, each `[count][width]`.
    fn message<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,layer:usize,spec:&LoopLayerDescription,which:Message,count:usize)->Result<(V,V)> {
        let (m,n,half)=self.message_form(run,layer,spec,which,count)?;
        need(!half,"a half-packed message norm requires the prefill gate path")?;
        Ok((m,n))
    }
    /// Steps 2 and 3 of a visit: the mix for attention, then the gates of a visit that reads messages.
    /// A layer with no attention query reads the stream as it is. One query gives one input for all three.
    /// With no messages the gates are skipped. They are not fed zeros.
    pub fn attention_inputs<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:&V)->Result<Inputs<V>> {
        self.attention_inputs_inner(run,visit,stream,false)
    }
    /// Like `attention_inputs`, also fusing `QueryNorm`/`KeyNorm`/`ValueNorm` into `MIX_ONE` (pass 1) or
    /// `GateBlend` (pass 2/3 when `count > 1`) when supported.
    pub(super) fn attention_inputs_normed<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:&V)->Result<Inputs<V>> {
        self.attention_inputs_inner(run,visit,stream,true)
    }
    fn attention_inputs_inner<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:&V,fuse_norms:bool)->Result<Inputs<V>> {
        let (layer,spec)=run.layer(visit)?;let c=run.compute;let width=run.width;let count=run.rows(stream)?;
        let rows=spec.attention_queries as usize;self.normed=None;
        let post_eps=run.description.epsilon as f32;
        let want_half=fuse_norms && count>super::SMALL_PASS_ROWS && run.weights.prefers_half_attention(layer);
        let msg_half=self.messages.as_ref().is_some_and(|m|m.2);
        // The gate norm of a mix, where the mix launch made it (`mix_rows_normed` or `Run::norm_blend_gate`).
        let mut gate_normed=None;
        let mut gate_normed_half=false;
        let inputs=if rows==0 {Inputs::Same(c.range(stream,0,count*width)?)} else {
            need(rows==1 || rows==ATTENTION_INPUTS,"the attention mix takes one query or three")?;
            let queries=run.weights.words(Weight::MixAttention(layer))?;need(queries.len()==rows*width,"the attention mix queries differ from the description")?;
            let transpose=rows>1 && count>1;
            let refs:Vec<&V>=self.stack.iter().collect();
            if fuse_norms && !visit.messages {
                if rows==1 && !spec.attention.separate_inputs {
                    let wt=run.weights.words(Weight::QueryNorm(layer))?;
                    if want_half && let Some((normed,_,post))=run.weights.mix_rows_normed_half(c,count,width,run.scale,MIX_NORM_EPSILON,post_eps,false,spec.boundary,false,queries,&refs,stream,Some(wt),false)? {
                        self.normed=normed;
                        return Ok(Inputs::NormedHalfSame(post));
                    }
                    if let Some((normed,_,post))=c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,post_eps,false,spec.boundary,false,queries,&refs,stream,Some(wt),false)? {
                        self.normed=normed;
                        return Ok(Inputs::NormedSame(post));
                    }
                } else if rows==ATTENTION_INPUTS {
                    if want_half {
                        let fused_h=if !spec.attention.separate_inputs {
                            let wt=run.weights.words(Weight::QueryNorm(layer))?;
                            run.weights.mix_rows_normed_half(c,count,width,run.scale,MIX_NORM_EPSILON,post_eps,transpose,spec.boundary,false,queries,&refs,stream,Some(wt),false)?
                        } else {
                            let mut qkv_w=Vec::with_capacity(ATTENTION_INPUTS*width);
                            qkv_w.extend_from_slice(run.weights.words(Weight::QueryNorm(layer))?);
                            qkv_w.extend_from_slice(run.weights.words(Weight::KeyNorm(layer))?);
                            qkv_w.extend_from_slice(run.weights.words(Weight::ValueNorm(layer))?);
                            run.weights.mix_rows_normed_half(c,count,width,run.scale,MIX_NORM_EPSILON,post_eps,transpose,spec.boundary,false,queries,&refs,stream,Some(&qkv_w),true)?
                        };
                        if let Some((normed,_,post))=fused_h {
                            self.normed=normed;
                            let part=count*(width/2);
                            return Ok(Inputs::NormedHalfEach([c.range(&post,0,part)?,c.range(&post,part,part)?,c.range(&post,2*part,part)?]));
                        }
                    }
                    let fused=if !spec.attention.separate_inputs {
                        let wt=run.weights.words(Weight::QueryNorm(layer))?;
                        c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,post_eps,transpose,spec.boundary,false,queries,&refs,stream,Some(wt),false)?
                    } else {
                        let mut qkv_w=Vec::with_capacity(ATTENTION_INPUTS*width);
                        qkv_w.extend_from_slice(run.weights.words(Weight::QueryNorm(layer))?);
                        qkv_w.extend_from_slice(run.weights.words(Weight::KeyNorm(layer))?);
                        qkv_w.extend_from_slice(run.weights.words(Weight::ValueNorm(layer))?);
                        c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,post_eps,transpose,spec.boundary,false,queries,&refs,stream,Some(&qkv_w),true)?
                    };
                    if let Some((normed,_,post))=fused {
                        self.normed=normed;
                        let part=count*width;
                        return Ok(Inputs::NormedEach([c.range(&post,0,part)?,c.range(&post,part,part)?,c.range(&post,2*part,part)?]));
                    }
                }
            }
            let (normed,mixed,gated,gated_h)=if visit.messages {
                if msg_half && fuse_norms && count>super::SMALL_PASS_ROWS
                    && let Some((normed,Some(mixed),post))=run.weights.mix_rows_normed_half(c,count,width,run.scale,MIX_NORM_EPSILON,post_eps,transpose,spec.boundary,true,queries,&refs,stream,None,false)?
                {
                    (normed,mixed,Some(post),true)
                } else if let Some((normed,Some(mixed),post))=c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,post_eps,transpose,spec.boundary,true,queries,&refs,stream,None,false)? {
                    (normed,mixed,Some(post),false)
                } else if rows==1 {
                    let (n,m,g)=run.norm_blend_gate(queries,&self.stack,stream,count,spec.boundary)?;
                    (n,m,g,false)
                } else {
                    let (normed,mixed)=run.norm_blend(queries,&self.stack,stream,count,transpose,spec.boundary)?;
                    (normed,mixed,None,false)
                }
            } else {
                let (normed,mixed)=run.norm_blend(queries,&self.stack,stream,count,transpose,spec.boundary)?;
                (normed,mixed,None,false)
            };
            gate_normed=gated;
            gate_normed_half=gated_h;
            // At the first layer of a block this norm is the row of the block before it (see `expert_input`).
            self.normed=normed;
            if rows==1 {Inputs::Same(mixed)} else {
                let part=count*width;
                Inputs::Each([c.range(&mixed,0,part)?,c.range(&mixed,part,part)?,c.range(&mixed,2*part,part)?])
            }
        };
        if !visit.messages {return Ok(inputs);}
        let order=[Message::Query,Message::Key,Message::Value];let mut gated=Vec::with_capacity(ATTENTION_INPUTS);
        let norm_weights=if fuse_norms {
            Some([run.weights.words(Weight::QueryNorm(layer))?,run.weights.words(Weight::KeyNorm(layer))?,run.weights.words(Weight::ValueNorm(layer))?])
        } else {None};
        match &inputs {
            // One norm serves the three gates: it is a function of the value alone.
            Inputs::Same(value)=>{
                if gate_normed_half && msg_half && let Some(nw)=norm_weights {
                    let normed=gate_normed.take().ok_or("loop decoder: missing half gate norm")?;
                    let msgs=order.into_iter().map(|which|self.message_form(run,layer,spec,which,count)).collect::<Result<Vec<_>>>()?;
                    if let (Some(i0),Some(i1),Some(i2))=(
                        run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,order[0]),&normed,&msgs[0].1,count)?,
                        run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,order[1]),&normed,&msgs[1].1,count)?,
                        run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,order[2]),&normed,&msgs[2].1,count)?,
                    ) {
                        let batch=[
                            (order[0],&i0,value,&msgs[0].0,Some(nw[0])),
                            (order[1],&i1,value,&msgs[1].0,Some(nw[1])),
                            (order[2],&i2,value,&msgs[2].0,Some(nw[2])),
                        ];
                        if want_half && let Some(normed_3)=run.weights.gate_outer_blend_norm(c,layer,&batch,count,post_eps,true)? {
                            let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused half gate result is absent".to_owned())?;
                            return Ok(Inputs::NormedHalfEach(out));
                        }
                        if let Some(normed_3)=run.weights.gate_outer_blend_norm(c,layer,&batch,count,post_eps,false)? {
                            let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused gate result is absent".to_owned())?;
                            return Ok(Inputs::NormedEach(out));
                        }
                    }
                    return Err("loop decoder: half-packed prefill gate path unavailable".into());
                }
                let normed=match gate_normed.take() {Some(normed)=>normed,None=>run.gate_norm(value)?};
                let messages=order.into_iter().map(|which|self.message(run,layer,spec,which,count)).collect::<Result<Vec<_>>>()?;
                if let Some(nw)=norm_weights {
                    if count>super::SMALL_PASS_ROWS
                        && let (Some(i0),Some(i1),Some(i2))=(
                            run.weights.product_rows_pair(c,Weight::GateDown(layer,order[0]),&normed,&messages[0].1,count)?,
                            run.weights.product_rows_pair(c,Weight::GateDown(layer,order[1]),&normed,&messages[1].1,count)?,
                            run.weights.product_rows_pair(c,Weight::GateDown(layer,order[2]),&normed,&messages[2].1,count)?,
                        )
                        && let Some(normed_3)=run.weights.gate_outer_blend_norm(c,layer,&[
                            (order[0],&i0,value,&messages[0].0,Some(nw[0])),
                            (order[1],&i1,value,&messages[1].0,Some(nw[1])),
                            (order[2],&i2,value,&messages[2].0,Some(nw[2])),
                        ],count,post_eps,false)?
                    {
                        let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused gate result is absent".to_owned())?;
                        return Ok(Inputs::NormedEach(out));
                    }
                    let pairs=run.gates_rows_pairs(layer,[&normed;3],&messages,count)?;
                    let batch=[(&pairs[0],value,&messages[0].0,Some(nw[0])),(&pairs[1],value,&messages[1].0,Some(nw[1])),(&pairs[2],value,&messages[2].0,Some(nw[2]))];
                    if let Some(normed_3)=c.gate_blend_norm(count,width,post_eps,&batch)? {
                        let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused gate result is absent".to_owned())?;
                        return Ok(Inputs::NormedEach(out));
                    }
                    for (pair,(m,_)) in pairs.iter().zip(&messages) {gated.push(run.gate_blend_only(pair,value,m)?);}
                } else {
                    let grouped=if count==1 {run.gates_one_row(layer,[value;3],[&normed;3],&messages)?} else {run.gates_rows(layer,[value;3],[&normed;3],&messages,count)?};
                    match grouped {
                        Some(all)=>gated=all,
                        None=>for (which,(m,n)) in order.into_iter().zip(&messages) {gated.push(run.gate(layer,which,value,&normed,(m,n))?);},
                    }
                }
            },
            // The three inputs of all rows: their gate norms are one launch (or fused into MIX_ONE), each row normed alone.
            Inputs::Each(values)=>{
                if gate_normed_half && msg_half && let Some(nw)=norm_weights {
                    let part=count*(width/2);
                    let normed=gate_normed.take().ok_or("loop decoder: missing half gate norm")?;
                    let normeds=(0..ATTENTION_INPUTS).map(|i|c.range(&normed,i*part,part)).collect::<Result<Vec<_>>>()?;
                    let msgs=order.into_iter().map(|which|self.message_form(run,layer,spec,which,count)).collect::<Result<Vec<_>>>()?;
                    if let (Some(i0),Some(i1),Some(i2))=(
                        run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,order[0]),&normeds[0],&msgs[0].1,count)?,
                        run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,order[1]),&normeds[1],&msgs[1].1,count)?,
                        run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,order[2]),&normeds[2],&msgs[2].1,count)?,
                    ) {
                        let batch=[
                            (order[0],&i0,&values[0],&msgs[0].0,Some(nw[0])),
                            (order[1],&i1,&values[1],&msgs[1].0,Some(nw[1])),
                            (order[2],&i2,&values[2],&msgs[2].0,Some(nw[2])),
                        ];
                        if want_half && let Some(normed_3)=run.weights.gate_outer_blend_norm(c,layer,&batch,count,post_eps,true)? {
                            let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused half gate result is absent".to_owned())?;
                            return Ok(Inputs::NormedHalfEach(out));
                        }
                        if let Some(normed_3)=run.weights.gate_outer_blend_norm(c,layer,&batch,count,post_eps,false)? {
                            let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused gate result is absent".to_owned())?;
                            return Ok(Inputs::NormedEach(out));
                        }
                    }
                    return Err("loop decoder: half-packed prefill gate path unavailable".into());
                }
                let part=count*width;
                let normed=match gate_normed.take() {Some(normed)=>normed,None=>run.gate_norm(&c.join(&values.iter().collect::<Vec<_>>())?)?};
                let normeds=(0..ATTENTION_INPUTS).map(|i|c.range(&normed,i*part,part)).collect::<Result<Vec<_>>>()?;
                let messages=order.into_iter().map(|which|self.message(run,layer,spec,which,count)).collect::<Result<Vec<_>>>()?;
                if let Some(nw)=norm_weights {
                    if count>super::SMALL_PASS_ROWS
                        && let (Some(i0),Some(i1),Some(i2))=(
                            run.weights.product_rows_pair(c,Weight::GateDown(layer,order[0]),&normeds[0],&messages[0].1,count)?,
                            run.weights.product_rows_pair(c,Weight::GateDown(layer,order[1]),&normeds[1],&messages[1].1,count)?,
                            run.weights.product_rows_pair(c,Weight::GateDown(layer,order[2]),&normeds[2],&messages[2].1,count)?,
                        )
                        && let Some(normed_3)=run.weights.gate_outer_blend_norm(c,layer,&[
                            (order[0],&i0,&values[0],&messages[0].0,Some(nw[0])),
                            (order[1],&i1,&values[1],&messages[1].0,Some(nw[1])),
                            (order[2],&i2,&values[2],&messages[2].0,Some(nw[2])),
                        ],count,post_eps,false)?
                    {
                        let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused gate result is absent".to_owned())?;
                        return Ok(Inputs::NormedEach(out));
                    }
                    let pairs=run.gates_rows_pairs(layer,[&normeds[0],&normeds[1],&normeds[2]],&messages,count)?;
                    let batch=[(&pairs[0],&values[0],&messages[0].0,Some(nw[0])),(&pairs[1],&values[1],&messages[1].0,Some(nw[1])),(&pairs[2],&values[2],&messages[2].0,Some(nw[2]))];
                    if let Some(normed_3)=c.gate_blend_norm(count,width,post_eps,&batch)? {
                        let out:[V;ATTENTION_INPUTS]=normed_3.try_into().map_err(|_|"loop decoder: a fused gate result is absent".to_owned())?;
                        return Ok(Inputs::NormedEach(out));
                    }
                    for ((pair,value),(m,_)) in pairs.iter().zip(values).zip(&messages) {gated.push(run.gate_blend_only(pair,value,m)?);}
                } else {
                    let grouped=if count==1 {run.gates_one_row(layer,[&values[0],&values[1],&values[2]],[&normeds[0],&normeds[1],&normeds[2]],&messages)?}
                        else {run.gates_rows(layer,[&values[0],&values[1],&values[2]],[&normeds[0],&normeds[1],&normeds[2]],&messages,count)?};
                    match grouped {
                        Some(all)=>gated=all,
                        None=>for (((which,value),normed),(m,n)) in order.into_iter().zip(values).zip(&normeds).zip(&messages) {gated.push(run.gate(layer,which,value,normed,(m,n))?);},
                    }
                }
            },
            Inputs::NormedSame(_)|Inputs::NormedEach(_)|Inputs::NormedHalfSame(_)|Inputs::NormedHalfEach(_)=>unreachable!("normed inputs only return when !visit.messages"),
        }
        let gated:[V;ATTENTION_INPUTS]=gated.try_into().map_err(|_|"loop decoder: a gate result is absent".to_owned())?;
        Ok(Inputs::Each(gated))
    }
    /// The list rule of step 7 and steps 8 and 9 of a visit: the input of the norm before the experts, 32-bit.
    /// `stream` is the stream that the visit read. `residual` is the stream after the attention half: in the first
    /// layer of a block the convolution output alone, in other layers `stream` plus that output.
    /// The first layer of a block puts `stream` on the block list. It is not added to the residual.
    pub fn expert_input<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:&V,residual:&V)->Result<V> {
        let (layer,spec)=run.layer(visit)?;let c=run.compute;let width=run.width;let count=run.rows(stream)?;
        need(c.words(residual)==count*width,"the residual differs from the rows of the stream")?;
        let normed=self.normed.take();
        if spec.boundary {self.stack.push(match normed {Some(row)=>row,None=>run.plain_norm(stream,count)?});}
        let query=run.weights.words(Weight::MixExperts(layer))?;need(query.len()==width,"the expert mix takes one query")?;
        if !visit.messages {return Ok(run.norm_blend(query,&self.stack,residual,count,false,false)?.1);}
        let (_,mixed,gated)=run.norm_blend_gate(query,&self.stack,residual,count,false)?;
        let (m,n)=self.message(run,layer,spec,Message::Expert,count)?;
        let normed=match gated {Some(normed)=>normed,None=>run.gate_norm(&mixed)?};
        run.gate(layer,Message::Expert,&mixed,&normed,(&m,&n))
    }
    /// Like `expert_input`, returning `(value, norm_form)`: fuses `ExpertNorm(layer)` into `MIX_ONE` (pass 1)
    /// or `GateBlend` (pass 2/3 when `count > 1`) when supported, eliminating separate `fast_norm` launches.
    pub(super) fn expert_input_normed<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:&V,residual:&V)->Result<(V,ExpertNorm)> {
        let (layer,spec)=run.layer(visit)?;let c=run.compute;let width=run.width;let count=run.rows(stream)?;
        need(c.words(residual)==count*width,"the residual differs from the rows of the stream")?;
        let normed=self.normed.take();
        if spec.boundary {self.stack.push(match normed {Some(row)=>row,None=>run.plain_norm(stream,count)?});}
        let query=run.weights.words(Weight::MixExperts(layer))?;need(query.len()==width,"the expert mix takes one query")?;
        let post_eps=run.description.epsilon as f32;
        let expert_w=run.weights.words(Weight::ExpertNorm(layer))?;
        let want_half=count>super::SMALL_PASS_ROWS && run.weights.prefers_half_experts(layer);
        if !visit.messages {
            let refs:Vec<&V>=self.stack.iter().collect();
            if want_half && let Some((_,_,z))=run.weights.mix_rows_normed_half(c,count,width,run.scale,MIX_NORM_EPSILON,post_eps,false,false,false,query,&refs,residual,Some(expert_w),false)? {
                return Ok((z,ExpertNorm::NormedHalf));
            }
            if let Some((_,_,z))=c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,post_eps,false,false,false,query,&refs,residual,Some(expert_w),false)? {
                return Ok((z,ExpertNorm::Normed));
            }
            return Ok((run.norm_blend(query,&self.stack,residual,count,false,false)?.1,ExpertNorm::Raw));
        }
        let msg_half=self.messages.as_ref().is_some_and(|m|m.2);
        let (_,mixed,gated,gated_h)={
            let refs:Vec<&V>=self.stack.iter().collect();
            if msg_half && count>super::SMALL_PASS_ROWS
                && let Some((_,Some(mixed),post))=run.weights.mix_rows_normed_half(c,count,width,run.scale,MIX_NORM_EPSILON,post_eps,false,false,true,query,&refs,residual,None,false)?
            {
                (None,mixed,Some(post),true)
            } else if let Some((_,Some(mixed),post))=c.mix_rows_normed(count,width,run.scale,MIX_NORM_EPSILON,post_eps,false,false,true,query,&refs,residual,None,false)? {
                (None,mixed,Some(post),false)
            } else {
                let (n,m,g)=run.norm_blend_gate(query,&self.stack,residual,count,false)?;
                (n,m,g,false)
            }
        };
        let (m,n,n_half)=self.message_form(run,layer,spec,Message::Expert,count)?;
        if gated_h && n_half && count>super::SMALL_PASS_ROWS {
            let normed=gated.ok_or("loop decoder: missing half expert gate norm")?;
            if let Some(inner)=run.weights.product_rows_pair_half_in(c,Weight::GateDown(layer,Message::Expert),&normed,&n,count)? {
                if want_half && let Some(mut outs)=run.weights.gate_outer_blend_norm(c,layer,&[(Message::Expert,&inner,&mixed,&m,Some(expert_w))],count,post_eps,true)? {
                    return Ok((outs.pop().ok_or_else(||"loop decoder: fused half expert gate output missing".to_owned())?,ExpertNorm::NormedHalf));
                }
                if let Some(mut outs)=run.weights.gate_outer_blend_norm(c,layer,&[(Message::Expert,&inner,&mixed,&m,Some(expert_w))],count,post_eps,false)? {
                    return Ok((outs.pop().ok_or_else(||"loop decoder: fused expert gate output missing".to_owned())?,ExpertNorm::Normed));
                }
            }
            return Err("loop decoder: half-packed expert gate path unavailable".into());
        }
        need(!gated_h && !n_half,"loop decoder: mismatched half gate norm state")?;
        let normed=match gated {Some(normed)=>normed,None=>run.gate_norm(&mixed)?};
        if count>super::SMALL_PASS_ROWS
            && let Some(inner)=run.weights.product_rows_pair(c,Weight::GateDown(layer,Message::Expert),&normed,&n,count)?
            && let Some(mut outs)=run.weights.gate_outer_blend_norm(c,layer,&[(Message::Expert,&inner,&mixed,&m,Some(expert_w))],count,post_eps,false)?
        {
            return Ok((outs.pop().ok_or_else(||"loop decoder: fused expert gate output missing".to_owned())?,ExpertNorm::Normed));
        }
        let pair=run.gate_pair_outer(layer,Message::Expert,count,&normed,(&m,&n))?;
        if let Some(mut outs)=c.gate_blend_norm(count,width,post_eps,&[(&pair,&mixed,&m,Some(expert_w))])? {
            return Ok((outs.pop().ok_or_else(||"loop decoder: fused expert gate output missing".to_owned())?,ExpertNorm::Normed));
        }
        Ok((run.gate_blend_only(&pair,&mixed,&m)?,ExpertNorm::Raw))
    }
    /// The mix before the last norm: every finished block and the stream after the last visit, with the output query.
    pub fn end_mix<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,stream:&V)->Result<V> {
        let count=run.rows(stream)?;
        let query=run.weights.words(Weight::MixOutput)?;need(query.len()==run.width,"the end mix takes one query")?;
        Ok(run.norm_blend(query,&self.stack,stream,count,false,false)?.1)
    }
    /// `end_mix` fused with weighted `LastNorm` when supported, eliminating the final `fast_norm` launch.
    pub(super) fn end_mix_normed<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,stream:&V)->Result<V> {
        let count=run.rows(stream)?;
        let query=run.weights.words(Weight::MixOutput)?;need(query.len()==run.width,"the end mix takes one query")?;
        let last_w=run.weights.words(Weight::LastNorm)?;
        let refs:Vec<&V>=self.stack.iter().collect();
        if let Some((_,_,normed))=run.compute.mix_rows_normed(count,run.width,run.scale,MIX_NORM_EPSILON,run.description.epsilon as f32,false,false,false,query,&refs,stream,Some(last_w),false)? {
            return Ok(normed);
        }
        let mixed=run.norm_blend(query,&self.stack,stream,count,false,false)?.1;
        run.rounded_norm(&mixed,last_w)
    }
    /// `end_mix` for the rows `picked` of the step only, in that order: `[picked][width]`. Each row of a mix reads its
    /// own vectors, so a row has the words of `end_mix`. The rows that want no end pay no end mix.
    pub fn end_mix_rows<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,stream:&V,picked:&[usize])->Result<V> {
        let (c,width,count)=(run.compute,run.width,run.rows(stream)?);
        need(!picked.is_empty() && picked.iter().all(|&row|row<count),"the end mix names a row outside the step")?;
        if picked.len()==count && picked.iter().enumerate().all(|(i,&row)|i==row) {return self.end_mix(run,stream);}
        let query=run.weights.words(Weight::MixOutput)?;need(query.len()==width,"the end mix takes one query")?;
        let row=run.plain_norm(&gather(c,stream,picked,width)?,picked.len())?;
        let rows=self.stack.iter().map(|entry|gather(c,entry,picked,width)).collect::<Result<Vec<_>>>()?;
        let mut list=rows.iter().collect::<Vec<_>>();list.push(&row);
        run.blend(query,&list,picked.len())
    }
}
