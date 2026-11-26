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
use super::{mix::{ExpertNorm,Run,Token,Weight,Weights,need,number,reciprocal,shape},rows::{Rows,rows_operation}};
use crate::{Result,bounded_decoder::{Compute,RouteFoldConv}};
use engine_model::{host_ops::router::RouterKind,plan::description::Visit,shapes::{ExpertShape,Ffn}};
use spark_gpu_probe::causal_ops::{Op,Shape};
#[cfg(test)] mod tests;

/// Form word of arm `ShortConv`: an addend is given, and the sum with the addend is rounded to BF16.
const CONV_ADDEND:u32=1<<1;
const CONV_ROUND:u32=1<<2;
/// Larger decode batches group the expert products by expert after reading the route.
const DEVICE_ROWS:usize=16;
/// The arithmetic form of the expert products. Decode always uses the row form, at every cohort width. Prefill
/// always uses the grouped tensor-core form where the device supports it, including a short final prompt slice.
/// A token therefore never changes arithmetic because unrelated chats joined or left its scheduler cohort.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) enum Phase {Prefill,Decode}
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
    sum_phase(run,visit,input,ExpertNorm::Raw,Phase::Prefill)
}
pub(super) fn sum_phase<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,input:&C::Value,norm:ExpertNorm,phase:Phase)->Result<C::Value> {
    let (layer,experts)=routed(run,visit)?;let c=run.compute;let w=run.weights;let r=run.description.routing;
    let z_half=norm==ExpertNorm::NormedHalf;
    let rows=if z_half {
        need(run.width().is_multiple_of(2) && c.words(input)>0 && c.words(input).is_multiple_of(run.width()/2),"a half expert input differs from the width")?;
        c.words(input)/(run.width()/2)
    } else {
        run.rows(input)?
    };
    let (count,picks)=(experts.count as usize,experts.selected as usize);
    need(r.kind==RouterKind::CorrectedSigmoid && !r.shared_gate && r.scale>0.0,"this router form is not built")?;
    let z=match norm {
        ExpertNorm::Raw=>input_norm(run,layer,input)?,
        ExpertNorm::Normed|ExpertNorm::NormedHalf=>c.range(input,0,c.words(input))?,
    };
    let logits=if z_half {
        w.product_rows_half_in(c,Weight::Router(layer),&z,rows)?.ok_or_else(||"loop decoder: half-input router unavailable".to_owned())?
    } else {
        w.product_rows(c,Weight::Router(layer),&z,rows)?
    };
    need(c.words(&logits)==rows*count,"the router product differs from the expert count")?;
    let bias=w.words(Weight::RouterBias(layer))?;need(bias.len()==count,"the router bias differs from the expert count")?;
    // Sigmoid scores, the pick by falling `score + bias`, and the weights `score / (sum + 1e-20) * scale` in pick order.
    let s=Shape {op:Op::Route,heads:number(r.groups as usize)?,dim:number(r.selected_groups as usize)?,taps:u32::from(r.normalize_selected),rows:number(count)?,cols:number(picks)?,epsilon:0.0,bound:r.scale as f32};
    // READ ARM 2: the rows pick in arm `RouterPick`, the first 2k words of `Op::Route` a row, bit for bit (host test in
    // block_experts_tests.rs): more rows in one rows launch, one row in the fast pick launch (W-SC2-POST: the words
    // read below are those 2k words only, the picks then their weights).
    let s=if spark_gpu_probe::block_rows::lengths(Shape {op:Op::RouterPick,..s},rows,0).is_some() {Shape {op:Op::RouterPick,..s}} else {s};
    let (route,compact)=if phase==Phase::Decode && rows>1 && rows<=spark_gpu_probe::fast_ops::ROUTER_PICK_COMPACT_ROWS_MAX {
        if let Some((route,compact))=c.router_pick_compact(s,rows,&logits,bias)? {
            (route,Some(compact))
        } else {
            (rows_operation(c,s,rows,[Some(&logits),None,None,None],0,bias)?,None)
        }
    } else {
        (rows_operation(c,s,rows,[Some(&logits),None,None,None],0,bias)?,None)
    };
    // Keep the base decode dispatch. Wide decode uses the grouped packed rows path below.
    if phase==Phase::Decode && rows<=DEVICE_ROWS {
        let (each,width)=(c.words(&route)/rows,run.width());
        need(each>=2*picks && c.words(&route)==rows*each,"the route of a step differs from its rows")?;
        let group=(spark_gpu_probe::packed_product::PICKS_MAX/picks.max(1)).max(1);
        // The picked numbers (`offset` 0) or the weights (`offset` picks) of rows `first..first + count`, row after row.
        let part=|first:usize,count:usize,offset:usize|->Result<C::Value> {
            if count==1 {return c.range(&route,first*each+offset,picks);}
            let parts=(first..first+count).map(|row|c.range(&route,row*each+offset,picks)).collect::<Result<Vec<_>>>()?;
            c.join(&parts.iter().collect::<Vec<_>>())
        };
        let mut sums=Vec::with_capacity(rows.div_ceil(group));
        for first in (0..rows).step_by(group) {
            let count=group.min(rows-first);
            let z_part=c.range(&z,first*width,count*width)?;
            let route_part=c.range(&route,first*each,count*each)?;
            let group_compact=if first==0 && count==rows {compact.as_ref()} else {None};
            if count>1 && let Some(sum)=device_sum_route(run,layer,experts,&z_part,&route_part,each,picks,count,group_compact)? {
                sums.push(sum);
                continue;
            }
            // No device form: the weights have none for any group, so the step takes the form for many rows.
            let Some(sum)=device_sum(run,layer,experts,&z_part,&part(first,count,0)?,&part(first,count,picks)?,picks,count)? else {break};
            sums.push(sum);
        }
        if sums.len()==rows.div_ceil(group) {
            if sums.len()==1 {return sums.pop().ok_or_else(||"loop decoder: a group of rows gave no expert sum".to_owned());}
            return c.join(&sums.iter().collect::<Vec<_>>());
        }
        need(sums.is_empty(),"loop decoder: the device form of the picked experts went away inside a step")?;
    }
    // Prefill and wide decode (`rows > DEVICE_ROWS`) group supported packed or widened expert products on the device.
    if let Some((weights,plan))=w.device_route_group(c,layer,&route,rows,picks,count)? {
        let (all,inner,width,caps)=(rows*picks,experts.width as usize,run.width(),run.description.caps);
        let cap_words=[caps.gate as f32,caps.up as f32,reciprocal(caps.gate),reciprocal(caps.up)];
        if z_half {
            let sum=w.device_routed_experts_half_in(c,layer,&plan,&z,cap_words,&weights,picks)?
                .ok_or_else(||"loop decoder: half-input routed experts unavailable".to_owned())?;
            need(c.words(&sum)==rows*width,"a routed expert sum differs from the rows of the step")?;
            return Ok(sum);
        }
        // The gated product, the down product and the fold in one call where the weights have it (the same words).
        if let Some(sum)=w.device_routed_experts(c,layer,&plan,&z,cap_words,&weights,picks)? {
            need(c.words(&sum)==rows*width,"a routed expert sum differs from the rows of the step")?;
            return Ok(sum);
        }
        // One launch for the gate-and-up product and its `CappedGate` where the weights have it (the same words).
        let gated=w.device_routed_gate(c,layer,&plan,&z,cap_words)?;
        let halves=if gated.is_some() {None} else {w.device_routed_products(c,Weight::GateUp(layer,0),&plan,&z)?};
        if gated.is_some() || halves.is_some() {
            let active=if let Some(active)=gated {active} else {
                let halves=halves.ok_or("loop decoder: the gate-and-up product went away")?;
                need(c.words(&halves)==all*2*inner,"a gate-and-up product differs from the expert width")?;
                let mut s=shape(Op::CappedGate);s.rows=number(all)?;s.dim=number(inner)?;s.epsilon=caps.gate as f32;s.bound=caps.up as f32;
                c.operation(s,[Some(&halves),None,None,None],&[reciprocal(caps.gate),reciprocal(caps.up)])?
            };
            need(c.words(&active)==all*inner,"a gated product differs from the expert width")?;
            // One call for the down product and its fold where the down stack has it (the same words, BF16 downs).
            if let Some(sum)=w.device_routed_down_fold(c,layer,&plan,&active,&weights,picks)? {
                need(c.words(&sum)==rows*width,"a routed down fold differs from the rows of the step")?;
                return Ok(sum);
            }
            let downs=w.device_routed_products(c,Weight::Down(layer,0),&plan,&active)?
                .ok_or("loop decoder: the down products of the routed experts have no device call")?;
            need(c.words(&downs)==all*width,"a down product differs from the width")?;
            let mut s=shape(Op::RouteFold);s.rows=number(picks)?;s.dim=number(width)?;
            return rows_operation(c,s,rows,[Some(&weights),Some(&downs),None,None],0,&[]);
        }
    }
    if z_half {return Err("loop decoder: half-input expert path requires device routing".into());}
    // Fallback (Host / Fixture / shapes outside device envelope): read router picks to host and group on CPU.
    // The words of each row are its picked numbers, then its weights in pick order. The weights go back as the
    // bits that were read: the host computes nothing on them.
    let words=c.read(&route)?;let each=words.len()/rows;
    need(each>=2*picks && words.len()==rows*each,"the route of a step differs from its rows")?;
    let ids=(0..rows).map(|row|picked(&words[row*each..row*each+picks],count)).collect::<Result<Vec<_>>>()?;
    let weights=c.upload(&(0..rows).flat_map(|row|words[row*each+picks..row*each+2*picks].iter().copied()).collect::<Vec<_>>())?;
    picked_sum(run,layer,experts,&z,&ids,&weights)
}
/// `picked_sum` of `rows` rows with the picked numbers left on the device (`Weights::picked_products`): the host reads
/// none, so the queue does not drain here. `z` is `[rows][width]`; `ids` and `weights` are `[rows][picks]`, the two
/// halves of the route of each row. All picks of the rows go in one launch a projection. The words are those of
/// `picked_sum` with the numbers read: the same products in pick order, the same gate, and the fold of each row in
/// pick order (with one row the reorder of `picked_sum` is the identity; with more, a product word reads only its own
/// row). None where the weights have no such call; the caller then reads the numbers. A down call that has none after
/// a gate-and-up call that had one is an error.
/// Strided-route `device_sum`: reads expert IDs and weights directly from `route` (`[rows][each]`) and maps
/// each pick `j` of `GateUp` directly to input row `j / picks` in `z`, avoiding both `part` joins and `spread`.
#[allow(clippy::too_many_arguments)]
fn device_sum_route<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layer:usize,experts:ExpertShape,z:&C::Value,route:&C::Value,each:usize,picks:usize,rows:usize,compact:Option<&C::Value>)->Result<Option<C::Value>> {
    let c=run.compute;let w=run.weights;let caps=run.description.caps;let (inner,width,all)=(experts.width as usize,run.width(),rows*picks);
    let cap_words=[caps.gate as f32,caps.up as f32,reciprocal(caps.gate),reciprocal(caps.up)];
    let repeat=if rows==1 {0} else {picks};
    let active=if let Some(active)=w.picked_products_strided_compact(c,Weight::GateUp(layer,0),route,rows,picks,each,z,repeat,Some(cap_words),compact)? {active} else {
        let Some(halves)=w.picked_products_strided_compact(c,Weight::GateUp(layer,0),route,rows,picks,each,z,repeat,None,compact)? else {return Ok(None)};
        need(c.words(&halves)==all*2*inner,"a gate-and-up product of the picked experts differs from the expert width")?;
        let mut s=shape(Op::CappedGate);s.rows=number(all)?;s.dim=number(inner)?;s.epsilon=caps.gate as f32;s.bound=caps.up as f32;
        c.operation(s,[Some(&halves),None,None,None],&cap_words[2..])?
    };
    need(c.words(&active)==all*inner,"a picked gate differs from the expert width")?;
    let downs=w.picked_products_strided_compact(c,Weight::Down(layer,0),route,rows,picks,each,&active,1,None,compact)?
        .ok_or("loop decoder: the down products of the picked experts have no call on the device")?;
    need(c.words(&downs)==all*width,"a down product of the picked experts differs from the width")?;
    if let Some(folded)=c.route_fold_strided(rows,picks,width,each,picks,route,&downs)? {return Ok(Some(folded));}
    let weights=if rows==1 {c.range(route,picks,picks)?} else {
        let parts=(0..rows).map(|row|c.range(route,row*each+picks,picks)).collect::<Result<Vec<_>>>()?;
        c.join(&parts.iter().collect::<Vec<_>>())?
    };
    let mut s=shape(Op::RouteFold);s.rows=number(picks)?;s.dim=number(width)?;
    rows_operation(c,s,rows,[Some(&weights),Some(&downs),None,None],0,&[]).map(Some)
}
#[allow(clippy::too_many_arguments)]
fn device_sum<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,layer:usize,experts:ExpertShape,z:&C::Value,ids:&C::Value,weights:&C::Value,picks:usize,rows:usize)->Result<Option<C::Value>> {
    let c=run.compute;let w=run.weights;let caps=run.description.caps;let (inner,width,all)=(experts.width as usize,run.width(),rows*picks);
    // One row: every pick reads `z` itself. More rows: the input row of each pick, in pick order.
    let each=if rows==1 {None} else {Some(run.spread(z,rows,picks,width)?)};
    let input=each.as_ref().unwrap_or(z);
    let cap_words=[caps.gate as f32,caps.up as f32,reciprocal(caps.gate),reciprocal(caps.up)];
    let active=if let Some(active)=w.picked_gate(c,layer,ids,input,rows==1,cap_words)? {active} else {
        let Some(halves)=w.picked_products(c,Weight::GateUp(layer,0),ids,input,rows==1,rows)? else {return Ok(None)};
        need(c.words(&halves)==all*2*inner,"a gate-and-up product of the picked experts differs from the expert width")?;
        let mut s=shape(Op::CappedGate);s.rows=number(all)?;s.dim=number(inner)?;s.epsilon=caps.gate as f32;s.bound=caps.up as f32;
        c.operation(s,[Some(&halves),None,None,None],&cap_words[2..])?
    };
    need(c.words(&active)==all*inner,"a picked gate differs from the expert width")?;
    let downs=w.picked_products(c,Weight::Down(layer,0),ids,&active,false,rows)?.ok_or("loop decoder: the down products of the picked experts have no call on the device")?;
    need(c.words(&downs)==all*width,"a down product of the picked experts differs from the width")?;
    let mut s=shape(Op::RouteFold);s.rows=number(picks)?;s.dim=number(width)?;
    rows_operation(c,s,rows,[Some(weights),Some(&downs),None,None],0,&[]).map(Some)
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
/// (`state::many::conv_rings`), `ring` is the window of the step and `step` one word. `residual` is the 32-bit stream
/// after the attention half, `[rows][width]`. The result is `[rows][width]`.
pub fn close<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,ring:&C::Value,step:&C::Value,window:u32,residual:&C::Value)->Result<C::Value> {
    let c=run.compute;let width=run.width();let taps=run.description.conv.kernel;let rows=run.rows(residual)?;
    let (ring_rows,steps)=if window==0 {(rows*taps as usize,rows)} else {(rows+taps as usize-1,1)};
    need(taps>0 && c.words(ring)==ring_rows*width && c.words(step)==steps,"the convolution ring after the experts differs from the description")?;
    let (s,weight)=close_shape(run,visit)?;
    rows_operation(c,s,rows,[Some(ring),None,Some(residual),Some(step)],window,weight)
}
fn close_shape<'a,C:Compute,W:Weights<C>>(run:&'a Run<'_,C,W>,visit:&Visit)->Result<(spark_gpu_probe::block_ops::Shape,&'a [f32])> {
    let (layer,_)=run.layer(visit)?;let width=run.width();let taps=run.description.conv.kernel;
    let weight=run.weights.words(Weight::ConvAfterExperts(layer))?;need(weight.len()==width*taps as usize,"the taps after the experts differ from the description")?;
    let mut s=shape(Op::ShortConv);s.heads=1;s.dim=number(width)?;s.taps=number(taps as usize)?;s.cols=CONV_ADDEND|CONV_ROUND;s.bound=conv_scale(taps+1);
    Ok((s,weight))
}
pub(super) fn close_rows<C:Compute,W:Weights<C>>(
    run:&Run<'_,C,W>,visit:&Visit,layout:&super::state::Layout,step:&Rows,
    states:&mut [&mut super::state::VisitState<C::Value>],sum:&C::Value,residual:&C::Value,
)->Result<C::Value> {
    need(run.rows(residual)?==step.total(),"the expert residual differs from the step")?;
    let (s,weight)=close_shape(run,visit)?;
    super::state::many::convolve(run.compute,layout,super::state::Ring::AfterExperts,step,states,s,sum,None,Some(residual),weight)
}
/// Combined expert half (`sum_phase` + `close_rows`) for multi-row steps. On kept decode passes (`rows = 2..=16`),
/// fuses `RouteFold` + `AfterExperts` `ShortConv` + `every` + `Residual` + `Round` into one launch when supported.
#[allow(clippy::too_many_arguments)]
pub(super) fn half_rows<C:Compute,W:Weights<C>>(
    run:&Run<'_,C,W>,visit:&Visit,layout:&super::state::Layout,step:&Rows,
    states:&mut [&mut super::state::VisitState<C::Value>],input:&C::Value,norm:ExpertNorm,residual:&C::Value,phase:Phase,
)->Result<C::Value> {
    let spans=step.chats();
    let separate=spans.len()>1 && spans.len()==step.total() && spans.len()<=16;
    if phase==Phase::Decode && norm!=ExpertNorm::NormedHalf && ((step.keeps() && spans.len()==1) || separate) {
        let (layer,experts)=routed(run,visit)?;
        let (c,w,r)=(run.compute,run.weights,run.description.routing);
        let (count,picks,rows,width)=(experts.count as usize,experts.selected as usize,step.total(),run.width());
        let (channels,taps)=layout.ring_size(super::state::Ring::AfterExperts);
        let group=(spark_gpu_probe::packed_product::PICKS_MAX/picks.max(1)).max(1);
        if (2..=DEVICE_ROWS).contains(&rows) && rows<=group && channels==width
            && (2..=spark_gpu_probe::fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps)
            && rows*picks<=spark_gpu_probe::fast_ops::ROUTE_FOLD_CONV_WEIGHTS_MAX
            && r.kind==RouterKind::CorrectedSigmoid && !r.shared_gate && r.scale>0.0
        {
            let z=if norm==ExpertNorm::Normed {c.range(input,0,c.words(input))?} else {input_norm(run,layer,input)?};
            let logits=w.product_rows(c,Weight::Router(layer),&z,rows)?;
            need(c.words(&logits)==rows*count,"the router product differs from the expert count")?;
            let bias=w.words(Weight::RouterBias(layer))?;
            need(bias.len()==count,"the router bias differs from the expert count")?;
            let s=Shape {op:Op::Route,heads:number(r.groups as usize)?,dim:number(r.selected_groups as usize)?,
                taps:u32::from(r.normalize_selected),rows:number(count)?,cols:number(picks)?,epsilon:0.0,bound:r.scale as f32};
            let s=if spark_gpu_probe::block_rows::lengths(Shape {op:Op::RouterPick,..s},rows,0).is_some() {Shape {op:Op::RouterPick,..s}} else {s};
            let (route,compact)=if let Some((route,compact))=c.router_pick_compact(s,rows,&logits,bias)? {
                (route,Some(compact))
            } else {
                (rows_operation(c,s,rows,[Some(&logits),None,None,None],0,bias)?,None)
            };
            let each=c.words(&route)/rows;
            need(each>=2*picks && c.words(&route)==rows*each,"the route of a step differs from its rows")?;
            let caps=run.description.caps;
            let (inner,all)=(experts.width as usize,rows*picks);
            let cap_words=[caps.gate as f32,caps.up as f32,reciprocal(caps.gate),reciprocal(caps.up)];
            let active=if let Some(active)=w.picked_products_strided_compact(c,Weight::GateUp(layer,0),&route,rows,picks,each,&z,picks,Some(cap_words),compact.as_ref())? {
                Some(active)
            } else if let Some(halves)=w.picked_products_strided_compact(c,Weight::GateUp(layer,0),&route,rows,picks,each,&z,picks,None,compact.as_ref())? {
                need(c.words(&halves)==all*2*inner,"a gate-and-up product of the picked experts differs from the expert width")?;
                let mut gs=shape(Op::CappedGate);gs.rows=number(all)?;gs.dim=number(inner)?;gs.epsilon=caps.gate as f32;gs.bound=caps.up as f32;
                Some(c.operation(gs,[Some(&halves),None,None,None],&cap_words[2..])?)
            } else {None};
            if let Some(active)=active {
                need(c.words(&active)==all*inner,"a picked gate differs from the expert width")?;
                let downs=w.picked_products_strided_compact(c,Weight::Down(layer,0),&route,rows,picks,each,&active,1,None,compact.as_ref())?
                    .ok_or("loop decoder: the down products of the picked experts have no call on the device")?;
                need(c.words(&downs)==all*width,"a down product of the picked experts differs from the width")?;
                let (cs,fixed)=close_shape(run,visit)?;
                let ring=super::state::Ring::AfterExperts as usize;
                let first_slot=(spans[0].first%taps as u64) as usize;
                let chat_rings=if separate {
                    states.iter().zip(spans).map(|(state,span)|(&state.rings[ring],(span.first%taps as u64) as usize)).collect()
                } else {Vec::new()};
                let req=RouteFoldConv {
                    rows, picks, dim:width, weight_stride:each, weight_offset:picks,
                    taps, first_slot, bound:cs.bound, half:false,
                    route:&route, downs:&downs, residual, fixed, ring:&states[0].rings[ring], chat_rings:&chat_rings,
                };
                if let Some((out,every))=c.route_fold_conv(&req)? {
                    let ring_words=taps*width;
                    need(c.words(&out)==rows*width && c.words(&every)==rows*ring_words,"the fused expert close differs from the layout")?;
                    if separate {
                        for (row,state) in states.iter_mut().enumerate() {
                            state.next[ring]=Some(c.range(&every,row*ring_words,ring_words)?);
                        }
                    } else {
                        states[0].next[ring]=Some(c.range(&every,(rows-1)*ring_words,ring_words)?);
                        states[0].every[ring]=Some(every);
                    }
                    return Ok(out);
                }
                let folded=if let Some(f)=c.route_fold_strided(rows,picks,width,each,picks,&route,&downs)? {f} else {
                    let parts=(0..rows).map(|row|c.range(&route,row*each+picks,picks)).collect::<Result<Vec<_>>>()?;
                    let weights=c.join(&parts.iter().collect::<Vec<_>>())?;
                    let mut rs=shape(Op::RouteFold);rs.rows=number(picks)?;rs.dim=number(width)?;
                    rows_operation(c,rs,rows,[Some(&weights),Some(&downs),None,None],0,&[])?
                };
                return close_rows(run,visit,layout,step,states,&folded,residual);
            }
        }
    }
    if phase==Phase::Prefill && !step.keeps() && spans.len()==1 {
        let (layer,experts)=routed(run,visit)?;
        let (c,w,r)=(run.compute,run.weights,run.description.routing);
        let (count,picks,rows,width)=(experts.count as usize,experts.selected as usize,step.total(),run.width());
        let (channels,taps)=layout.ring_size(super::state::Ring::AfterExperts);
        if rows>=taps && rows>1 && (2..=spark_gpu_probe::fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps)
            && channels==width && r.kind==RouterKind::CorrectedSigmoid && !r.shared_gate && r.scale>0.0
        {
            let z_half=norm==ExpertNorm::NormedHalf;
            let z=match norm {
                ExpertNorm::Raw=>input_norm(run,layer,input)?,
                ExpertNorm::Normed|ExpertNorm::NormedHalf=>c.range(input,0,c.words(input))?,
            };
            let logits=if z_half {
                w.product_rows_half_in(c,Weight::Router(layer),&z,rows)?.ok_or_else(||"loop decoder: half-input router unavailable".to_owned())?
            } else {
                w.product_rows(c,Weight::Router(layer),&z,rows)?
            };
            need(c.words(&logits)==rows*count,"the router product differs from the expert count")?;
            let bias=w.words(Weight::RouterBias(layer))?;
            need(bias.len()==count,"the router bias differs from the expert count")?;
            let s=Shape {op:Op::Route,heads:number(r.groups as usize)?,dim:number(r.selected_groups as usize)?,
                taps:u32::from(r.normalize_selected),rows:number(count)?,cols:number(picks)?,epsilon:0.0,bound:r.scale as f32};
            let s=if spark_gpu_probe::block_rows::lengths(Shape {op:Op::RouterPick,..s},rows,0).is_some() {Shape {op:Op::RouterPick,..s}} else {s};
            let route=rows_operation(c,s,rows,[Some(&logits),None,None,None],0,bias)?;
            if let Some((weights,plan))=w.device_route_group(c,layer,&route,rows,picks,count)? {
                let caps=run.description.caps;
                let cap_words=[caps.gate as f32,caps.up as f32,reciprocal(caps.gate),reciprocal(caps.up)];
                let (cs,fixed)=close_shape(run,visit)?;
                let ring_idx=super::state::Ring::AfterExperts as usize;
                let first_slot=(spans[0].first%taps as u64) as usize;
                let closed=if z_half {
                    w.device_routed_experts_close_half_in(
                        c,layer,&plan,&z,cap_words,&weights,picks,residual,fixed,&states[0].rings[ring_idx],taps,first_slot,cs.bound,
                    )?
                } else {
                    w.device_routed_experts_close(
                        c,layer,&plan,&z,cap_words,&weights,picks,residual,fixed,&states[0].rings[ring_idx],taps,first_slot,cs.bound,
                    )?
                };
                if let Some((out,next_ring))=closed {
                    need(c.words(&out)==rows*width && c.words(&next_ring)==taps*width,"the prefill fused expert close differs from the layout")?;
                    states[0].next[ring_idx]=Some(next_ring);
                    return Ok(out);
                }
                let summed=if z_half {
                    w.device_routed_experts_half_in(c,layer,&plan,&z,cap_words,&weights,picks)?
                } else {
                    w.device_routed_experts(c,layer,&plan,&z,cap_words,&weights,picks)?
                };
                if let Some(sum)=summed {
                    need(c.words(&sum)==rows*width,"a routed expert sum differs from the rows of the step")?;
                    return close_rows(run,visit,layout,step,states,&sum,residual);
                }
            }
        }
    }
    let sum=sum_phase(run,visit,input,norm,phase)?;
    close_rows(run,visit,layout,step,states,&sum,residual)
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
