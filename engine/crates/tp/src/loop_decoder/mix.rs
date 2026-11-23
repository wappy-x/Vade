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
    /// The 32-bit words of a small weight, in the layout that `Weight` gives.
    fn words(&self,weight:Weight)->Result<&[f32]>;
    /// The matrix times one input row. Each output word is one chain over the columns in rising order from +0.0, a
    /// product and its add are two rounded 32-bit steps, and the sum is not rounded. `chain_product` is the host form.
    fn product(&self,compute:&C,weight:Weight,x:&C::Value)->Result<C::Value>;
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
    /// `products` on the experts that the route picked, in pick order, with their numbers left where `ids` is: `ids`
    /// holds the picked numbers as whole 32-bit floats. `first` is expert 0 of the projection and layer
    /// (`Weight::GateUp(layer,0)` or `Weight::Down(layer,0)`). The words are those of `products` on the weights of
    /// those numbers. None: no such call here; the caller reads the numbers and calls `products`.
    fn picked_products(&self,_compute:&C,_first:Weight,_ids:&C::Value,_x:&C::Value,_shared:bool)->Result<Option<C::Value>> {Ok(None)}
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

/// The inputs of the attention norms of one visit. They are 32-bit and are not normed yet.
#[derive(Debug)]
pub enum Inputs<V> {
    /// One vector is the query, the key and the value input.
    Same(V),
    /// Query, key, value.
    Each([V;ATTENTION_INPUTS]),
}
/// What a half of a visit works with: the backend, the weights and the description of the checkpoint.
/// `tables` holds the index table of each swap of a step on the backend, by its two counts: a swap of a step of
/// rows uploads its table once a step, not once a call (`Run::swap`; `clear_tables` at the start of each step).
pub struct Run<'a,C:Compute,W:Weights<C>> {pub compute:&'a C,pub weights:&'a W,pub description:&'a LoopTextDescription,width:usize,scale:f32,
    tables:std::cell::RefCell<std::collections::BTreeMap<(usize,usize),C::Value>>}
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
        Ok(Self {compute,weights,description,width,scale:mix_scale(width),tables:Default::default()})
    }
    /// `rows::swap` with the index table of its two counts uploaded once and kept until `clear_tables`.
    pub(super) fn swap(&self,x:&C::Value,first:usize,second:usize,width:usize)->Result<C::Value> {
        if first==1 || second==1 {return swap(self.compute,x,first,second,width);}
        let mut tables=self.tables.borrow_mut();
        let table=match tables.entry((first,second)) {
            std::collections::btree_map::Entry::Occupied(kept)=>kept.into_mut(),
            std::collections::btree_map::Entry::Vacant(new)=>new.insert(self.compute.upload(&swap_table(first,second))?),
        };
        gather_table(self.compute,x,table,first*second,width)
    }
    /// Let the index tables of the swaps go: a step of another count of rows has other tables.
    pub(super) fn clear_tables(&self) {self.tables.borrow_mut().clear();}
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
    /// One gate for each row of a step: `(1 - tanh(F)) * value + tanh(G) * message`, 32-bit. Every value here is
    /// `[rows][width]`. `normed` is `gate_norm(value)` and `message` is the message with its gate norm,
    /// `(message, gate_norm(message))`: the norms of all messages of a pass are one launch (`Token::enter`).
    /// The inner vector is the down matrix times the two rounded norms side by side, rounded to BF16. `F` and `G` are
    /// the forget and the gain matrix times the inner vector, 32-bit. Arm `GateBlend` makes the two products and the sum.
    fn gate(&self,layer:usize,which:Message,value:&C::Value,normed:&C::Value,(message,message_normed):(&C::Value,&C::Value))->Result<C::Value> {
        let c=self.compute;let w=self.weights;let (gate_width,width)=(self.description.core.gate_rank as usize,self.width);
        let count=self.rows(value)?;
        need([normed,message,message_normed].iter().all(|v|c.words(v)==count*width),"a gate input differs from the rows of the step")?;
        // Two values side by side for each row of the step.
        let pair=|first:&C::Value,second:&C::Value|->Result<C::Value> {let joined=c.join(&[first,second])?;if count==1 {Ok(joined)} else {self.swap(&joined,2,count,width)}};
        let inner=w.product_rows(c,Weight::GateDown(layer,which),&pair(normed,message_normed)?,count)?;
        need(c.words(&inner)==count*gate_width,"a gate's inner product differs from the gate rank")?;
        let inner=c.round(&inner)?;
        // One row: the forget and the gain matrix on the same inner vector are one products call; where the owner holds
        // them one after another the two parts lie side by side and the pair below is a view.
        let (forget,gain)=if count==1 {
            let mut y=w.stack_products(c,&[Weight::GateForget(layer,which),Weight::GateGain(layer,which)],&inner)?.into_iter();
            (y.next().ok_or("loop decoder: a forget product is missing")?,y.next().ok_or("loop decoder: a gain product is missing")?)
        } else {(w.product_rows(c,Weight::GateForget(layer,which),&inner,count)?,w.product_rows(c,Weight::GateGain(layer,which),&inner,count)?)};
        need(c.words(&forget)==count*width && c.words(&gain)==count*width,"a gate's outer product differs from the width")?;
        let mut s=shape(Op::GateBlend);s.dim=number(width)?;
        rows_operation(c,s,count,[Some(&pair(&forget,&gain)?),Some(value),Some(message),None],0,&[])
    }
}

/// The mix state of the rows of one step: it lives for one step and holds no chat state. A token step is a step of
/// one row. Every value that goes in or comes out is `[rows][width]`, row-major, and a row reads its own words only.
/// The stack holds the plain norm of each vector of the block list, oldest first: the embedding, then each finished
/// block. Each entry is `[rows][width]`. `messages` holds the messages of the running pass and their gate norms, each
/// `[messages][rows][width]`, so one message of all rows is one range.
pub struct Token<V> {stack:Vec<V>,normed:Option<V>,start:Option<(V,usize)>,messages:Option<(V,V)>}
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
        let last=run.plain_norm(&stream,count)?;
        let queries=run.weights.words(Weight::Messages)?;
        need(queries.len()==(d.core.end-d.core.start) as usize*MESSAGES*width,"the message queries differ from the loop range")?;
        // No visit reads the messages of the pass before again: they go before the new ones are made.
        self.messages=None;
        // The mix gives `[rows][messages][width]`.
        let messages={let mut from=self.stack[*kept+1..].iter().collect::<Vec<_>>();from.push(&last);run.swap(&run.blend(queries,&from,count)?,count,queries.len()/width,width)?};
        // The gate norm of every message of the pass and every row in one launch: each row is normed alone.
        let normed=run.gate_norm(&messages)?;
        let restart=c.range(saved,0,count*width)?;
        self.stack.truncate(*kept);self.normed=None;self.messages=Some((messages,normed));
        Ok(restart)
    }
    /// One message of a layer for the `count` rows of the step and its gate norm, each `[count][width]`.
    fn message<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,layer:usize,spec:&LoopLayerDescription,which:Message,count:usize)->Result<(V,V)> {
        let start=run.description.core.start as usize;
        need(spec.message_gates && layer>=start,"a visit with messages names a layer with no gates")?;
        let (all,normed)=self.messages.as_ref().ok_or("loop decoder: a visit reads messages that no pass made")?;
        let (at,len)=(((layer-start)*MESSAGES+which as usize)*count*run.width,count*run.width);
        Ok((run.compute.range(all,at,len)?,run.compute.range(normed,at,len)?))
    }
    /// Steps 2 and 3 of a visit: the mix for attention, then the gates of a visit that reads messages.
    /// A layer with no attention query reads the stream as it is. One query gives one input for all three.
    /// With no messages the gates are skipped. They are not fed zeros.
    pub fn attention_inputs<C:Compute<Value=V>,W:Weights<C>>(&mut self,run:&Run<'_,C,W>,visit:&Visit,stream:&V)->Result<Inputs<V>> {
        let (layer,spec)=run.layer(visit)?;let c=run.compute;let width=run.width;let count=run.rows(stream)?;
        let rows=spec.attention_queries as usize;self.normed=None;
        let inputs=if rows==0 {Inputs::Same(c.range(stream,0,count*width)?)} else {
            need(rows==1 || rows==ATTENTION_INPUTS,"the attention mix takes one query or three")?;
            let queries=run.weights.words(Weight::MixAttention(layer))?;need(queries.len()==rows*width,"the attention mix queries differ from the description")?;
            let normed=run.plain_norm(stream,count)?;
            let mixed={let mut list=self.stack.iter().collect::<Vec<_>>();list.push(&normed);run.blend(queries,&list,count)?};
            // At the first layer of a block this norm is the row of the block before it (see `expert_input`).
            self.normed=Some(normed);
            if rows==1 {Inputs::Same(mixed)} else {
                // `[rows of the step][3][width]` to one value of all rows for each input.
                let (each,part)=(run.swap(&mixed,count,rows,width)?,count*width);
                Inputs::Each([c.range(&each,0,part)?,c.range(&each,part,part)?,c.range(&each,2*part,part)?])
            }
        };
        if !visit.messages {return Ok(inputs);}
        let order=[Message::Query,Message::Key,Message::Value];let mut gated=Vec::with_capacity(ATTENTION_INPUTS);
        match &inputs {
            // One norm serves the three gates: it is a function of the value alone.
            Inputs::Same(value)=>{let normed=run.gate_norm(value)?;for which in order {let (m,n)=self.message(run,layer,spec,which,count)?;gated.push(run.gate(layer,which,value,&normed,(&m,&n))?);}},
            // The three inputs of all rows: their gate norms are one launch, each row normed alone.
            Inputs::Each(values)=>{
                let (normed,part)=(run.gate_norm(&c.join(&values.iter().collect::<Vec<_>>())?)?,count*width);
                for (i,(which,value)) in order.into_iter().zip(values).enumerate() {let (m,n)=self.message(run,layer,spec,which,count)?;gated.push(run.gate(layer,which,value,&c.range(&normed,i*part,part)?,(&m,&n))?);}
            },
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
        let row=run.plain_norm(residual,count)?;
        let mixed={let mut list=self.stack.iter().collect::<Vec<_>>();list.push(&row);run.blend(query,&list,count)?};
        if !visit.messages {return Ok(mixed);}
        let (m,n)=self.message(run,layer,spec,Message::Expert,count)?;
        run.gate(layer,Message::Expert,&mixed,&run.gate_norm(&mixed)?,(&m,&n))
    }
    /// The mix before the last norm: every finished block and the stream after the last visit, with the output query.
    pub fn end_mix<C:Compute<Value=V>,W:Weights<C>>(&self,run:&Run<'_,C,W>,stream:&V)->Result<V> {
        let count=run.rows(stream)?;
        let query=run.weights.words(Weight::MixOutput)?;need(query.len()==run.width,"the end mix takes one query")?;
        let row=run.plain_norm(stream,count)?;let mut list=self.stack.iter().collect::<Vec<_>>();list.push(&row);
        run.blend(query,&list,count)
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
