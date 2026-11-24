//! The expert half of one visit behind one function: the router, the picked experts, their sum, and the last
//! stream sum. Only this file knows the stored form of the experts.
//!
//! `half` is that function: steps 7 (the block list) to 12 of a visit. It is made of three parts, so that the state
//! file can write the convolution ring between them: `Token::expert_input` (the mix and the gate), `sum` (the norm,
//! the router and the experts) and `close` (the convolution after the experts and the stream after the visit).
//!
//! This file is written over the trait `Compute`, so its host run is the byte reference of its device run.
//! It follows these lines of the reference (`transformers` v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`)
//! and holds its own code: the weighted norm, 730-745; the soft caps and the activation, 751-771; one expert and the
//! sum of the picked experts, 787-811; the router, 1152-1172; the convolution after the experts and the stream after
//! the visit, 285-393 and 906-910.
//!
//! Number types: the input of the norm is 32-bit. The normed input, each expert product, the activation and the
//! sum of the picked experts are BF16 values. The router is 32-bit. The stream after the visit is rounded to BF16 once.
//!
//! Orders that the reference leaves open, fixed here for every path: the pick is in falling `score + bias` with the
//! lower expert number first on equal values, and the picked experts add in that order (`Op::Route`, arm `RouteFold`).
use super::{mix::{Run,Token,Weight,Weights,need,number,reciprocal,shape},rows::rows_operation};
use crate::{Result,bounded_decoder::Compute};
use engine_model::{host_ops::router::RouterKind,plan::description::Visit,shapes::{ExpertShape,Ffn}};
use spark_gpu_probe::causal_ops::{Op,Shape};
#[cfg(test)] mod tests;

/// Form word of arm `ShortConv`: an addend is given, and the sum with the addend is rounded to BF16.
const CONV_ADDEND:u32=1<<1;
const CONV_ROUND:u32=1<<2;
/// The most rows of a step whose expert sum runs row by row with the picks on the device (`device_sum`). Such a
/// step reads an expert once a pick; a larger step reads each picked expert once, one launch an expert, after one
/// read of the route. Measured on the decode of 1 to 8 chats (the row form is the faster one there).
const DEVICE_ROWS:usize=8;
/// The number that the convolution after the experts multiplies by, for `terms` terms (its taps and the input that is
/// added back). The reference divides by `sqrt(terms)` (line 392): see `reciprocal`.
pub fn conv_scale(terms:u64)->f32 {reciprocal((terms as f64).sqrt())}
/// The routed experts of the layer of a visit.
fn routed<'a,C:Compute,W:Weights<C>>(run:&Run<'a,C,W>,visit:&Visit)->Result<(usize,ExpertShape)> {
    let (layer,spec)=run.layer(visit)?;
    let Ffn::Experts(experts)=spec.ffn else {return Err("loop decoder: the expert half needs routed experts".into());};
    need(experts.shared_width==0 && experts.count>0 && experts.selected>0 && experts.selected<=experts.count && experts.width>0,"the expert shape is not built")?;
    Ok((layer,experts))
}
/// The norm before the experts: weighted, with the config's epsilon, rounded to BF16 once (arm `RoundedNorm`).
/// This is the one place where this file norms.
fn input_norm<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layer:usize,input:&C::Value)->Result<C::Value> {
    let weight=run.weights.words(Weight::ExpertNorm(layer))?;need(weight.len()==run.width(),"the expert norm weight differs from the width")?;
    run.rounded_norm(input,weight)
}
/// The expert numbers that `Op::Route` wrote as 32-bit words: each a whole number under `count`, each once.
fn picked(words:&[f32],count:usize)->Result<Vec<usize>> {
    let mut ids=Vec::with_capacity(words.len());
    for &word in words {
        need(word>=0.0 && word.fract()==0.0 && (word as usize)<count,"the router picked no expert of this layer")?;
        need(!ids.contains(&(word as usize)),"the router picked one expert twice")?;ids.push(word as usize);
    }Ok(ids)
}
/// Step 10 of a visit, for the rows of a step: the norm, the router and the picked experts. `input` is the 32-bit
/// result of `Token::expert_input`, `[rows][width]`. The result is the sum of the picked experts of each row before
/// its convolution, `[rows][width]`, BF16 values. A row reads its own words only.
/// The host reads the picked expert numbers here, one read a visit for all rows: it chooses the weights of the products.
pub fn sum<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,input:&C::Value)->Result<C::Value> {
    let (layer,experts)=routed(run,visit)?;let c=run.compute;let w=run.weights;let r=run.description.routing;
    let (count,picks,rows)=(experts.count as usize,experts.selected as usize,run.rows(input)?);
    need(r.kind==RouterKind::CorrectedSigmoid && !r.shared_gate && r.scale>0.0,"this router form is not built")?;
    let z=input_norm(run,layer,input)?;
    let logits=w.product_rows(c,Weight::Router(layer),&z,rows)?;need(c.words(&logits)==rows*count,"the router product differs from the expert count")?;
    let bias=w.words(Weight::RouterBias(layer))?;need(bias.len()==count,"the router bias differs from the expert count")?;
    // Sigmoid scores, the pick by falling `score + bias`, and the weights `score / (sum + 1e-20) * scale` in pick order.
    let s=Shape {op:Op::Route,heads:number(r.groups as usize)?,dim:number(r.selected_groups as usize)?,taps:u32::from(r.normalize_selected),rows:number(count)?,cols:number(picks)?,epsilon:0.0,bound:r.scale as f32};
    // READ ARM 2: more rows pick in one rows launch of arm `RouterPick`, the first 2k words of `Op::Route` a row, bit
    // for bit (host test in block_experts_tests.rs); one row keeps `Op::Route`.
    let s=if rows>1 && spark_gpu_probe::block_rows::lengths(Shape {op:Op::RouterPick,..s},rows,0).is_some() {Shape {op:Op::RouterPick,..s}} else {s};
    let route=rows_operation(c,s,rows,[Some(&logits),None,None,None],0,bias)?;
    if rows==1 {
        let (ids,weights)=(c.range(&route,0,picks)?,c.range(&route,picks,picks)?);
        if let Some(sum)=device_sum(run,layer,experts,&z,&ids,&weights,picks)? {return Ok(sum);}
        let ids=picked(&c.read(&ids)?,count)?;
        return picked_sum(run,layer,experts,&z,&[ids],&weights);
    }
    // A step of a few rows (the decode of several chats, one new token each): each row runs the form of one row with
    // its picks left on the device, so the host reads no route and the queue does not drain here. A row's words do
    // not depend on the other rows of the step, so they are those of `picked_sum` for all rows.
    if rows<=DEVICE_ROWS {
        let (each,width)=(c.words(&route)/rows,run.width());
        need(each>=2*picks && c.words(&route)==rows*each,"the route of a step differs from its rows")?;
        let mut sums=Vec::with_capacity(rows);
        for row in 0..rows {
            let (ids,weights)=(c.range(&route,row*each,picks)?,c.range(&route,row*each+picks,picks)?);
            // No device form: the weights have none for any row, so the step takes the form for many rows.
            let Some(sum)=device_sum(run,layer,experts,&c.range(&z,row*width,width)?,&ids,&weights,picks)? else {break};
            sums.push(sum);
        }
        if sums.len()==rows {return c.join(&sums.iter().collect::<Vec<_>>());}
        need(sums.is_empty(),"loop decoder: the device form of the picked experts went away inside a step")?;
    }
    // The words of each row are its picked numbers, then its weights in pick order. The weights go back as the
    // bits that were read: the host computes nothing on them.
    let words=c.read(&route)?;let each=words.len()/rows;
    need(each>=2*picks && words.len()==rows*each,"the route of a step differs from its rows")?;
    let ids=(0..rows).map(|row|picked(&words[row*each..row*each+picks],count)).collect::<Result<Vec<_>>>()?;
    let weights=c.upload(&(0..rows).flat_map(|row|words[row*each+picks..row*each+2*picks].iter().copied()).collect::<Vec<_>>())?;
    picked_sum(run,layer,experts,&z,&ids,&weights)
}
/// `picked_sum` of one row with the picked numbers left on the device (`Weights::picked_products`): the host reads
/// none, so the queue does not drain here. `ids` and `weights` are the two halves of the route of the row. The words
/// are those of `picked_sum` with the numbers read: the same products in pick order, the same gate, and the fold in
/// pick order (with one row the reorder of `picked_sum` is the identity). None where the weights have no such call;
/// the caller then reads the numbers. A down call that has none after a gate-and-up call that had one is an error.
fn device_sum<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layer:usize,experts:ExpertShape,z:&C::Value,ids:&C::Value,weights:&C::Value,picks:usize)->Result<Option<C::Value>> {
    let c=run.compute;let w=run.weights;let caps=run.description.caps;let (inner,width)=(experts.width as usize,run.width());
    let Some(halves)=w.picked_products(c,Weight::GateUp(layer,0),ids,z,true)? else {return Ok(None)};
    need(c.words(&halves)==picks*2*inner,"a gate-and-up product of the picked experts differs from the expert width")?;
    let mut s=shape(Op::CappedGate);s.rows=number(picks)?;s.dim=number(inner)?;s.epsilon=caps.gate as f32;s.bound=caps.up as f32;
    let active=c.operation(s,[Some(&halves),None,None,None],&[reciprocal(caps.gate),reciprocal(caps.up)])?;
    let downs=w.picked_products(c,Weight::Down(layer,0),ids,&active,false)?.ok_or("loop decoder: the down products of the picked experts have no call on the device")?;
    need(c.words(&downs)==picks*width,"a down product of the picked experts differs from the width")?;
    let mut s=shape(Op::RouteFold);s.rows=number(picks)?;s.dim=number(width)?;
    rows_operation(c,s,1,[Some(weights),Some(&downs),None,None],0,&[]).map(Some)
}
/// THE SEAM OF THE EXPERT FORM. From the normed input `z` (`[rows][width]`), the picked expert numbers of each row
/// and their weights (`[rows][picks]`, in pick order) to the summed expert output, `[rows][width]`. Nothing outside
/// this function knows how the expert numbers are made. A second form replaces the body of this function and
/// nothing else.
///
/// With one row each pick reads `z`, in pick order. With more rows the picks of the step run in the order of the
/// expert number: one product launch for an expert takes all rows that picked it, so each expert matrix is read
/// one time a visit. A product word is one chain over its own input row, and the sum of a row adds its picks in
/// pick order, so the order of the launches changes no word.
///
/// The form here reads each expert as BF16 weights that the load made, and BF16 inputs:
/// 1. For each pick, the gate-and-up matrix times `z`.
/// 2. Arm `CappedGate`: `rn((cap(gate, a) * sigmoid(gate)) * cap(up, b))` with `cap(x, c) = tanh(x * r) * c`
///    and `r` the `reciprocal` of the cap (the reference divides by the cap, lines 757 and 768-770). The sigmoid reads the
///    gate that is not capped. The arm rounds each product word to BF16 as it reads it. Its weight is two words: the
///    reciprocal of the gate cap, then of the up cap.
/// 3. For each pick, the down matrix times its activation.
/// 4. Arm `RouteFold`: `rn(sum over the picks, in pick order, of down * weight)`, with each down word rounded to
///    BF16 as it is read. One BF16 round after the sum.
fn picked_sum<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layer:usize,experts:ExpertShape,z:&C::Value,ids:&[Vec<usize>],weights:&C::Value)->Result<C::Value> {
    let c=run.compute;let w=run.weights;let caps=run.description.caps;let (rows,inner,width)=(ids.len(),experts.width as usize,run.width());
    let picks=ids.first().map_or(0,Vec::len);let all=rows*picks;
    need(picks>0 && ids.iter().all(|row|row.len()==picks) && c.words(weights)==all && c.words(z)==rows*width,"the route weights differ from the picks")?;
    let join=|parts:&[C::Value]|c.join(&parts.iter().collect::<Vec<_>>());
    // The picks in the order of their launches, as (row, expert). One row: the pick order. More rows: by expert.
    let mut order=(0..all).collect::<Vec<_>>();
    if rows>1 {order.sort_by_key(|&pick|(ids[pick/picks][pick%picks],pick));}
    let expert=|at:usize|ids[order[at]/picks][order[at]%picks];
    // The runs of one expert in that order: its number, its first pick and its count of picks.
    let mut runs:Vec<(usize,usize,usize)>=Vec::new();
    for at in 0..all {match runs.last_mut() {Some(last) if rows>1 && last.0==expert(at)=>last.2+=1,_=>runs.push((expert(at),at,1))}}
    // One row (a token step): the products of all picks in one call where the weights have it, the same words as one
    // call a pick. More rows: the runs read the input row of each pick through a table (`Weights::product_runs`), in
    // launch order.
    let grouped=|kind:fn(usize,usize)->Weight,x:&C::Value,shared:bool|->Result<Option<C::Value>> {
        if rows!=1 {return Ok(None);}
        w.products(c,&runs.iter().map(|&(number,_,_)|kind(layer,number)).collect::<Vec<_>>(),x,shared)
    };
    let launches=|kind:fn(usize,usize)->Weight|runs.iter().map(|&(number,_,count)|(kind(layer,number),count)).collect::<Vec<_>>();
    let halves=if rows==1 {match grouped(Weight::GateUp,z,true)? {Some(all)=>all,None=>join(&runs.iter().map(|&(number,_,_)|w.product(c,Weight::GateUp(layer,number),z)).collect::<Result<Vec<_>>>()?)?}} else {
        let input=order.iter().map(|&pick|pick/picks).collect::<Vec<_>>();
        w.product_runs(c,&launches(Weight::GateUp),z,width,Some(&input),None)?
    };
    need(c.words(&halves)==all*2*inner,"a gate-and-up product differs from the expert width")?;
    let mut s=shape(Op::CappedGate);s.rows=number(all)?;s.dim=number(inner)?;s.epsilon=caps.gate as f32;s.bound=caps.up as f32;
    let active=c.operation(s,[Some(&halves),None,None,None],&[reciprocal(caps.gate),reciprocal(caps.up)])?;
    // The downs back in row order, each row with its picks in pick order: row `pick` of the result is launch row
    // `place[pick]`. One row: the launch order is the pick order.
    let mut place=vec![0usize;all];for (at,&pick) in order.iter().enumerate() {place[pick]=at;}
    let downs=match grouped(Weight::Down,&active,false)? {Some(all)=>all,None=>w.product_runs(c,&launches(Weight::Down),&active,inner,None,if rows==1 {None} else {Some(&place)})?};
    need(c.words(&downs)==all*width,"a down product differs from the width")?;
    let mut s=shape(Op::RouteFold);s.rows=number(picks)?;s.dim=number(width)?;
    rows_operation(c,s,rows,[Some(weights),Some(&downs),None,None],0,&[])
}
/// Steps 11 and 12 of a visit, for the rows of a step: `f = rn((the taps on the ring + u) * conv_scale)`, then the
/// stream after the visit, `rn(residual + f)`. One launch of arm `ShortConv` with the addend and the round.
/// `ring` holds the convolution ring of each row, `[rows][taps][width]`, with the expert sum `u` of that row in its
/// slot (`state::row_rings`). `step` is one word a row: that slot. Or, with `window` the mask of the window form
/// (`state::conv_rings`), `ring` is the window of the step and `step` one word. `residual` is the 32-bit stream
/// after the attention half, `[rows][width]`. The result is `[rows][width]`.
pub fn close<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,ring:&C::Value,step:&C::Value,window:u32,residual:&C::Value)->Result<C::Value> {
    let (layer,_)=run.layer(visit)?;let c=run.compute;let width=run.width();let taps=run.description.conv.kernel;let rows=run.rows(residual)?;
    let (ring_rows,steps)=if window==0 {(rows*taps as usize,rows)} else {(rows+taps as usize-1,1)};
    need(taps>0 && c.words(ring)==ring_rows*width && c.words(step)==steps,"the convolution ring after the experts differs from the description")?;
    let weight=run.weights.words(Weight::ConvAfterExperts(layer))?;need(weight.len()==width*taps as usize,"the taps after the experts differ from the description")?;
    let mut s=shape(Op::ShortConv);s.heads=1;s.dim=number(width)?;s.taps=number(taps as usize)?;s.cols=CONV_ADDEND|CONV_ROUND;s.bound=conv_scale(taps+1);
    rows_operation(c,s,rows,[Some(ring),None,Some(residual),Some(step)],window,weight)
}
/// The expert half of one visit. `stream` is the stream that the visit read. `residual` is the stream after the
/// attention half. `ring` puts the expert sum of this token into the convolution ring of this visit and gives the
/// ring and its slot word (see `close`). The result is the stream after the visit, BF16 values.
pub fn half<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,token:&mut Token<C::Value>,stream:&C::Value,residual:&C::Value,
    ring:impl FnOnce(&C::Value)->Result<(C::Value,C::Value)>)->Result<C::Value> {
    let input=token.expert_input(run,visit,stream,residual)?;
    let (ring,step)=ring(&sum(run,visit,&input)?)?;
    close(run,visit,&ring,&step,0,residual)
}
