//! The attention half of one visit: steps 4 to 7. It takes the inputs of the attention norms and the stream that
//! the visit read, and gives the stream after the first short convolution.
//!
//! This file is written over the trait `Compute`, so its host run is the byte reference of its device run.
//! It follows these lines of the reference (`transformers` v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`)
//! and holds its own code: the scores, the sink and the weighted values, 111-142; the clips and the position scale,
//! 176-236; the relative term, 239-270; the convolutions, 285-455; the projections and the output gate, 458-543;
//! the two layer kinds, 546-727; the weighted norm, 730-745; the place of the norms, of the convolution after
//! attention and of the first sum of the stream in a visit, 876-905.
//!
//! The launches of one visit, in order. An arm is an operation of `causal_ops` with that name.
//! 1. The norms of the inputs: arm `RoundedNorm` with a weight. One launch when one vector is all three inputs and
//!    the layer has one attention norm, else three.
//! 2. Six products on the normed inputs: query, relative state, sink and output gate read the query input; the key
//!    reads the key input and the value reads the value input. A layer that makes taps from the token adds two.
//! 3. The query: arm `HeadClip` with the learned scale, and with the position factor in a layer that has one.
//! 4. The relative state: `HeadClip` in the same form. Arm `RelBias` gives the term of each head and distance (in
//!    step 7, unless the backend runs the one-launch form).
//! 5. The key: its product goes into the key ring of the state, arm `ShortConv` reads the ring, `HeadClip` with no
//!    scale clips each key head, and the finished key goes packed into its row of the key plane.
//! 6. The value: its product goes into the value ring, `ShortConv`, then `RoundedNorm` with one row a key head and
//!    no weight, and the finished value goes packed into its row of the value plane.
//!    A backend with `Compute::conv_finish` runs step 5 and step 6 as one launch each, with the same words.
//! 7. Arm `WindowScores`, arm `SinkSoftmax`, arm `WindowValues` (it holds the output gate). They read every key
//!    and value from the state, those of the current token too. A backend with `Compute::window_one` runs the
//!    relative term and these three arms as one launch with the same words.
//! 8. The output product. It goes into the ring after attention. `ShortConv` reads the ring and adds the stream
//!    that the visit read. The first layer of a block adds nothing: its residual is the convolution output alone.
//!
//! Number types. A product gives its 32-bit sums and is not rounded. Each arm that reads one rounds each word as it
//! reads it, so the words are those of "round, then use" and no round launch runs. A ring holds the product as it
//! is. The results of the arms are BF16 values. The residual is 32-bit and is not rounded; in the first layer of a
//! block it is the BF16 output of the convolution.
//!
//! Every number that this file makes from the config is made in 64-bit and rounded to 32-bit once: `score_scale`,
//! `position_factor`, the clip thresholds, and the reciprocal of each convolution (`conv_scale` of the expert file).
use super::{experts::conv_scale,mix::{Inputs,Run,Weight,Weights,need,number,reciprocal,shape},state::{self,Layout,Ring,VisitState,size}};
use crate::{Result,bounded_decoder::{Compute,ConvFinish,WindowOne,WindowSplit}};
use engine_model::plan::description::{ConvTaps,PositionScale,Visit};
use spark_gpu_probe::{block_stream::{CONV_ADDEND,CONV_TOKEN_TAPS},causal_ops::{Op,Shape},fixed_math::ln};

/// The number that a score is multiplied by: the 32-bit value of `head size ^ (-1/2)`, made in 64-bit and rounded
/// once (reference line 126). It is not a BF16 value: with the factor rounded to BF16 first, fewer words of the
/// lab trace equal the reference (the lab ruling of 2026-11-22 14:02 on the score scale).
pub fn score_scale(head_size:u64)->f32 {reciprocal((head_size as f64).sqrt())}
/// The factor of the queries and of the relative states of the token at `position`, in a layer that has one:
/// `1 + alpha * ln(max((position + 1) / floor, 1))` (reference lines 216-218).
/// The reference divides a 32-bit tensor by the number `floor`. That is a product with the reciprocal (lab ruling
/// R8, receipt `divform.json` of the reference answers). Each step is one rounded 32-bit step, and `ln` is the
/// shared one. The whole-model host form of the model crate has the same steps (`sink_attention`, `Reach`).
pub fn position_factor(scale:PositionScale,position:u64)->f32 {
    let ratio=(position.saturating_add(1) as f32*reciprocal(scale.floor as f64)).max(1.0);
    1.0+scale.alpha as f32*ln(ratio)
}
/// A weighted norm with a rounded result: `rn((x * scale) * weight)`, or `rn(x * scale)` with no weight, for each of
/// `rows` rows. `scale = 1 / sqrt(mean of squares + epsilon)`, with the squares added in the tree of the norm
/// kernel. The input is read as it is. This is the one norm call of this half and of the token step.
pub fn rounded_norm<C:Compute>(c:&C,x:&C::Value,rows:usize,width:usize,epsilon:f64,weight:Option<&[f32]>)->Result<C::Value> {
    need(rows>0 && width>0 && c.words(x)==rows*width && weight.is_none_or(|weight|weight.len()==width),"a norm input or weight differs from its width")?;
    let mut s=shape(Op::RoundedNorm);s.heads=number(rows)?;s.dim=number(width)?;s.taps=u32::from(weight.is_some());s.epsilon=epsilon as f32;
    c.operation(s,[Some(x),None,None,None],weight.unwrap_or(&[]))
}
/// Arm `HeadClip` on `heads` heads of `dim` values: the learned scale if one is given, the clip at `threshold`,
/// one round, and with a factor the product with it and one more round.
fn head_clip<C:Compute>(c:&C,x:&C::Value,heads:usize,dim:usize,scale:&[f32],factor:Option<&C::Value>,threshold:f64)->Result<C::Value> {
    need(threshold>0.0 && c.words(x)==heads*dim && (scale.is_empty() || scale.len()==heads*dim),"a clip input or scale differs from the description")?;
    let mut s=shape(Op::HeadClip);s.heads=number(heads)?;s.dim=number(dim)?;s.taps=u32::from(!scale.is_empty());s.cols=u32::from(factor.is_some());s.bound=threshold as f32;
    c.operation(s,[Some(x),factor,None,None],scale)
}
/// The convolution of the keys or of the values. `product` is the key or value product of the current token: it
/// goes into its ring first, so the arm reads every tap from the ring. `made` holds the taps that a projection
/// made from the token, in a layer that has them. The result is rounded: BF16 values.
#[allow(clippy::too_many_arguments)]
fn stream_conv<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layout:&Layout,rings:&mut [C::Value;4],position:u64,which:Ring,fixed:Weight,
    made:Option<&C::Value>,heads:usize,product:&C::Value)->Result<C::Value> {
    let c=run.compute;let (channels,taps)=layout.ring_size(which);
    let fixed=run.weights.words(fixed)?;
    need(channels.is_multiple_of(heads) && fixed.len()==channels*taps && made.is_none_or(|made|c.words(made)==heads*taps),"the taps of a key or value convolution differ from the description")?;
    let step=state::write_ring(c,layout,which,rings,position,product)?;
    c.operation(conv_shape(heads,channels,taps,made.is_some())?,[Some(&rings[which as usize]),made,None,Some(&step)],fixed)
}
/// The shape of arm `ShortConv` of the keys or the values: `heads` heads of `channels / heads` values, `taps` taps.
fn conv_shape(heads:usize,channels:usize,taps:usize,made:bool)->Result<Shape> {
    let mut s=shape(Op::ShortConv);s.heads=number(heads)?;s.dim=number(channels/heads)?;s.taps=number(taps)?;
    // The divisor counts each tap once, the made taps once more, and the input that is added back
    // (reference lines 392 and 454).
    s.cols=if made {CONV_TOKEN_TAPS} else {0};s.bound=conv_scale(if made {2*taps as u64+1} else {taps as u64+1});
    Ok(s)
}
/// The key or the value of one token: its ring, its fixed taps, the taps made from the token, its product, and the
/// arm that finishes it (`HeadClip` with no scale at threshold `param`, or `RoundedNorm` with no weight at epsilon
/// `param`).
struct Finish<'a,V> {which:Ring,fixed:Weight,made:Option<&'a V>,product:&'a V,op:Op,param:f64}
/// Step 5 for the key or the value of the token at `position`: the product goes into its ring, arm `ShortConv` reads
/// the ring, the clip or the norm finishes each head, and the finished row goes packed into its row of the plane.
/// A backend with `Compute::conv_finish` runs these as one launch with the same words.
fn finish_row<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layout:&Layout,state:&mut VisitState<C::Value>,position:u64,heads:usize,head:usize,
    f:&Finish<'_,C::Value>)->Result<Option<C::Value>> {
    let c=run.compute;let (channels,taps)=layout.ring_size(f.which);
    let fixed=run.weights.words(f.fixed)?;
    need(channels==heads*head && channels==layout.kv_channels && fixed.len()==channels*taps && f.made.is_none_or(|made|c.words(made)==heads*taps)
        && c.words(f.product)==channels && f.param>0.0,"the taps of a key or value convolution differ from the description")?;
    let plane=if f.which==Ring::Key {&mut state.keys} else {&mut state.values};
    let cap=state::plane_rows(c,layout,plane)?;
    let slot=layout.slot(position)?;
    need(slot<cap,"a key or value row differs from the layout")?;
    let step=state::step_words(c,&[layout.ring_slot(f.which,position)])?;
    let one=ConvFinish {shape:conv_shape(heads,channels,taps,f.made.is_some())?,finish:f.op,param:f.param as f32,fixed,made:f.made,step:&step,
        product:f.product,plane_word:slot*layout.row_words};
    // The one-launch form gives its finished row (no reader here): the caller keeps it while the next finish records.
    if let Some(row)=c.conv_finish(&one,&mut state.rings[f.which as usize],plane)? {return Ok(Some(row));}
    let y=stream_conv(run,layout,&mut state.rings,position,f.which,f.fixed,f.made,heads,f.product)?;
    let finished=if f.op==Op::HeadClip {head_clip(c,&y,heads,head,&[],None,f.param)?} else {rounded_norm(c,&y,heads,head,f.param,None)?};
    let plane=if f.which==Ring::Key {&mut state.keys} else {&mut state.values};
    state::write_row(c,layout,plane,position,&finished)?;Ok(None)
}
/// Steps 4 to 7 of one visit. `keep` is the stream that the visit read. `inputs` are the inputs of the attention
/// norms, from `Token::attention_inputs`. `state` is the state of this visit and `layout` its sizes; the token at
/// `position` writes its rows into the state here, before the attention reads the state.
/// The result is the residual inside the block: `keep` plus the convolution output, 32-bit and not rounded. In the
/// first layer of a block it is the convolution output alone.
/// A step that fails has written only rows of `position`. No token up to that position can see what those rows
/// held before, so the same step can run again on the same state.
pub fn half<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,layout:&Layout,state:&mut VisitState<C::Value>,position:u64,
    keep:&C::Value,inputs:&Inputs<C::Value>)->Result<C::Value> {
    let (layer,spec)=run.layer(visit)?;let (c,w,a,width,epsilon)=(run.compute,run.weights,spec.attention,run.width(),run.description.epsilon);
    let (heads,kv_heads,head,relative,extent)=(size(a.query_heads)?,size(a.kv_heads)?,size(a.head_dim)?,size(a.relative.dim)?,size(a.relative.extent)?);
    need(heads>0 && kv_heads>0 && heads.is_multiple_of(kv_heads) && head>0 && head.is_multiple_of(2) && relative>0 && extent>0 && epsilon>0.0,"the attention shape is not built")?;
    need(layout.kv_channels==kv_heads*head && layout.channels==width && layout.ring==a.window.is_some() && layout.kv_taps as u64==a.key_value_conv.kernel
        && layout.taps as u64==run.description.conv.kernel,"the state layout differs from the layer of the visit")?;
    need(c.words(keep)==width,"the stream differs from the width")?;
    // Step 4. A layer with one attention norm gives the same weight under all three names.
    let norm=|x:&C::Value,rows:usize,which:Weight|rounded_norm(c,x,rows,width,epsilon,Some(w.words(which)?));
    let normed=match inputs {
        Inputs::NormedSame(x)=>vec![c.range(x,0,c.words(x))?],
        Inputs::NormedEach([q,k,v])=>vec![c.range(q,0,c.words(q))?,c.range(k,0,c.words(k))?,c.range(v,0,c.words(v))?],
        Inputs::NormedHalfSame(_)|Inputs::NormedHalfEach(_)=>return Err("loop decoder: single-row attention takes FP32 inputs".into()),
        Inputs::Same(x) if !a.separate_inputs=>vec![norm(x,1,Weight::QueryNorm(layer))?],
        Inputs::Same(x)=>vec![norm(x,1,Weight::QueryNorm(layer))?,norm(x,1,Weight::KeyNorm(layer))?,norm(x,1,Weight::ValueNorm(layer))?],
        // Three inputs and one weight (a sliding layer that reads messages): one launch of three rows. A row is
        // normed alone, so the words are those of three launches.
        Inputs::Each([q,k,v]) if !a.separate_inputs=>{
            let all=norm(&c.join(&[q,k,v])?,3,Weight::QueryNorm(layer))?;
            vec![c.range(&all,0,width)?,c.range(&all,width,width)?,c.range(&all,2*width,width)?]
        },
        Inputs::Each([q,k,v])=>vec![norm(q,1,Weight::QueryNorm(layer))?,norm(k,1,Weight::KeyNorm(layer))?,norm(v,1,Weight::ValueNorm(layer))?],
    };
    let (xq,xk,xv)=(&normed[0],&normed[normed.len()/2],&normed[normed.len()-1]);
    let product=|which:Weight,x:&C::Value,rows:usize|->Result<C::Value> {
        let y=w.product(c,which,x)?;need(c.words(&y)==rows,"an attention product differs from the description")?;Ok(y)
    };
    // Step 5, the products on the query input: with one input for all three, the key and the value too. One call
    // (`Weights::products`): an owner of weights that holds them one after another runs them as one product.
    let one=normed.len()==1;
    let names=[Weight::Query(layer),Weight::Relative(layer),Weight::Sink(layer),Weight::OutputGate(layer),Weight::Key(layer),Weight::Value(layer)];
    let sizes=[heads*head,heads*relative,heads,heads*head,kv_heads*head,kv_heads*head];
    let made_xq=w.stack_products(c,&names[..if one {6} else {4}],xq)?;
    need(made_xq.iter().zip(sizes).all(|(y,n)|c.words(y)==n),"an attention product differs from the description")?;
    let mut made_xq=made_xq.into_iter();let mut next=||made_xq.next().ok_or("loop decoder: an attention product is missing".to_owned());
    let (query,moved,sink,gate)=(next()?,next()?,next()?,next()?);
    // The key or the value product, and in a layer that makes taps from the token the product of those taps on the
    // same input: one products call.
    let with_taps=|main:Weight,taps:Weight,x:&C::Value|->Result<(C::Value,Option<C::Value>)> {
        if matches!(a.key_value_conv.taps,ConvTaps::Fixed {..}) {return Ok((product(main,x,kv_heads*head)?,None));}
        let mut y=w.stack_products(c,&[main,taps],x)?.into_iter();
        let (main,taps)=(y.next().ok_or("loop decoder: a key or value product is missing")?,y.next().ok_or("loop decoder: a tap product is missing")?);
        need(c.words(&main)==kv_heads*head && c.words(&taps)==kv_heads*layout.kv_taps,"an attention product differs from the description")?;Ok((main,Some(taps)))
    };
    let made=|which:Weight,x:&C::Value|->Result<Option<C::Value>> {
        match a.key_value_conv.taps {ConvTaps::Fixed {..}=>Ok(None),ConvTaps::FixedAndToken {..}=>product(which,x,kv_heads*layout.kv_taps).map(Some)}
    };
    let ((key_product,key_made),(value_product,value_made))=if one {
        ((next()?,made(Weight::KeyTapProjection(layer),xk)?),(next()?,made(Weight::ValueTapProjection(layer),xv)?))
    } else {(with_taps(Weight::Key(layer),Weight::KeyTapProjection(layer),xk)?,with_taps(Weight::Value(layer),Weight::ValueTapProjection(layer),xv)?)};
    // Step 5, the query side. The position factor is one word for the queries and the relative states of the token.
    let factor=match a.position_scale {Some(scale)=>Some(c.upload_constant(&[position_factor(scale,position)])?),None=>None};
    let (query_scale,moved_scale)=(w.words(Weight::QueryClip(layer))?,w.words(Weight::RelativeClip(layer))?);
    let table=w.words(Weight::RelativeTable(layer))?;need(table.len()==relative*extent,"the distance table differs from the description")?;
    // Step 5, the key and the value of this token. Each goes into the state before the attention reads the state.
    let key=Finish {which:Ring::Key,fixed:Weight::KeyTaps(layer),made:key_made.as_ref(),product:&key_product,op:Op::HeadClip,param:a.query_key_clip};
    // The key's finished row of the one-launch form lives until the value's finish is recorded, so the two launches
    // write apart (no reuse of its words) and the recorder may run them as one (`conv_norm::pair`). No one reads it.
    let key_row=finish_row(run,layout,state,position,kv_heads,head,&key)?;
    let value=Finish {which:Ring::Value,fixed:Weight::ValueTaps(layer),made:value_made.as_ref(),product:&value_product,op:Op::RoundedNorm,param:epsilon};
    finish_row(run,layout,state,position,kv_heads,head,&value)?;drop(key_row);
    // Step 5, the attention. The views of the planes live only inside this block: the next token writes the planes.
    // A backend with the one-launch form (`Compute::window_one`) gives the words of the four arms below in one launch;
    // otherwise arm `RelBias` gives the term of each head and distance, then the scores, the softmax and the values.
    let attended={
        let plane_cap=state::plane_rows(c,layout,&state.keys)?;
        let (rows,live,group)=(layout.view_rows_in(position,plane_cap)?,layout.live(position)?,heads/kv_heads);
        let step=state::step_words(c,&[live,layout.slot(position)?])?;
        let (keys,values)=(state::view(c,layout,&state.keys,rows)?,state::view(c,layout,&state.values,rows)?);
        let one=WindowOne {rows:1,heads,kv_heads,view:rows,extent,scale:score_scale(a.head_dim),query:&query,query_scale,query_bound:a.query_key_clip as f32,
            moved:&moved,moved_scale,moved_bound:a.relative.clip as f32,position:factor.as_ref(),keys:&keys,values:&values,written:None,rel_table:table,sink:&sink,gate:&gate,steps:&step};
        let split_req = WindowSplit {
            base: WindowOne {rows:one.rows,heads:one.heads,kv_heads:one.kv_heads,view:one.view,extent:one.extent,scale:one.scale,
                query:one.query,query_scale:one.query_scale,query_bound:one.query_bound,moved:one.moved,moved_scale:one.moved_scale,
                moved_bound:one.moved_bound,position:one.position,keys:one.keys,values:one.values,written:None,rel_table:one.rel_table,
                sink:one.sink,gate:one.gate,steps:one.steps},
            is_ring: layout.ring, first_pos: position, row_pos_live: &[(position, live)], delta: None, planes: &[],
        };
        let fast = match c.window_split(&split_req)? {
            Some(attended) => Some(attended),
            None => if live<=spark_gpu_probe::flash_window::FW_ONE_VIEW_MAX {c.window_one(&one)?} else {None},
        };
        match fast {
            Some(attended)=>{need(c.words(&attended)==heads*head,"the one-launch attention differs from the description")?;attended},
            None=>{
                let query=head_clip(c,&query,heads,head,query_scale,factor.as_ref(),a.query_key_clip)?;
                let moved=head_clip(c,&moved,heads,relative,moved_scale,factor.as_ref(),a.relative.clip)?;
                let mut s=shape(Op::RelBias);s.heads=number(heads)?;s.dim=number(relative)?;s.cols=number(extent)?;
                let bias=c.operation(s,[Some(&moved),None,None,None],table)?;
                let mut s=shape(Op::WindowScores);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(group)?;s.rows=number(rows)?;s.cols=number(extent)?;s.bound=score_scale(a.head_dim);
                let scores=c.operation(s,[Some(&query),Some(&keys),Some(&bias),Some(&step)],&[])?;
                let mut s=shape(Op::SinkSoftmax);s.heads=number(heads)?;s.rows=number(rows)?;
                let weights=c.operation(s,[Some(&scores),Some(&sink),Some(&state::step_words(c,&[live])?),None],&[])?;
                let mut s=shape(Op::WindowValues);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(group)?;s.rows=number(rows)?;
                c.operation(s,[Some(&weights),Some(&values),Some(&gate),Some(&step)],&[])?
            },
        }
    };
    // Steps 6 and 7. The output product goes into its ring; one launch gives the convolution and the first sum.
    let output=product(Weight::Output(layer),&attended,width)?;
    let step=state::write_ring(c,layout,Ring::AfterAttention,&mut state.rings,position,&output)?;
    let taps=w.words(Weight::ConvAfterAttention(layer))?;need(taps.len()==width*layout.taps,"the taps after attention differ from the description")?;
    let mut s=shape(Op::ShortConv);s.heads=1;s.dim=number(width)?;s.taps=number(layout.taps)?;s.bound=conv_scale(layout.taps as u64+1);
    let ring=&state.rings[Ring::AfterAttention as usize];
    if spec.boundary {return c.operation(s,[Some(ring),None,None,Some(&step)],taps);}
    s.cols=CONV_ADDEND;c.operation(s,[Some(ring),None,Some(keep),Some(&step)],taps)
}
pub mod many;
pub use many::half_rows;
#[cfg(test)]
pub(super) mod hand;
#[cfg(test)]
mod tests;
