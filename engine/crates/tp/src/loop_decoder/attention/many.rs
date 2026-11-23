//! The attention half of one visit for a step of many rows (`rows::Rows`): piece D1 of the rows work. A row is one
//! position of one chat.
//!
//! The launches are those of `half`, in its order, each for all rows of the step. Every value is row-major:
//! `[rows][words of one row]`. A row gets the input words that it gets in a step of one row and runs the same arm,
//! so its words are those of `half`. Rows meet in three places, each inside one chat:
//! - The key and the value convolution, and the convolution after attention: a row reads the inputs of the
//!   positions before it. Each row gets a ring of its own by one gather (`state::row_rings`).
//! - The planes: the finished keys and values of all rows go into the planes first. Then each row reads the keys
//!   of its chat up to its own position: its live count and its slot say which.
//! - So the three launches over keys (`WindowScores`, `SinkSoftmax`, `WindowValues`) run once for each chat, with
//!   the plane of that chat as an input that all its rows share.
//!
//! The part over keys (the relative term, the scores, the softmax, the weighted values) runs in groups of rows of
//! one chat, so that the scores of one group stay under the byte limit of the layout (`Layout::key_bytes`). A group changes no word: no row reads a score
//! of another row.
//!
//! No ring of the state is written here. The rings after the step are pending in the state until
//! `state::commit`. The planes are written: only slots of the positions of this step, which no row of a
//! position before the step can see.
use super::{position_factor,rounded_norm,score_scale};
use super::super::{experts::conv_scale,mix::{Inputs,Run,Weight,Weights,need,number,shape},rows::{Rows,rows_operation,shared},state::{self,Layout,Ring,VisitState,size,many::{key_steps,room,row_rings,write_rows}}};
use crate::{Result,bounded_decoder::Compute};
use engine_model::plan::description::{ConvTaps,Visit};
use spark_gpu_probe::{block_stream::{CONV_ADDEND,CONV_TOKEN_TAPS},causal_ops::Op};

/// Steps 4 to 7 of one visit for the rows of a step. `states` holds the state of this visit of each chat, in step
/// order. `keep` is the stream that the visit read, `[rows][width]`. `inputs` are the inputs of the attention
/// norms, each `[rows][width]`. The result is the residual inside the block, `[rows][width]`, 32-bit and not
/// rounded. With one chat and one row it gives the words of `half`: a row has the words of `half` at its position
/// after the rows before it in its chat.
pub fn half_rows<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,layout:&Layout,step:&Rows,states:&mut [&mut VisitState<C::Value>],
    keep:&C::Value,inputs:&Inputs<C::Value>)->Result<C::Value> {
    let (spans,key_bytes)=(step.chats(),layout.key_bytes);
    let (layer,spec)=run.layer(visit)?;let (c,w,a,width,epsilon)=(run.compute,run.weights,spec.attention,run.width(),run.description.epsilon);
    let (heads,kv_heads,head,relative,extent)=(size(a.query_heads)?,size(a.kv_heads)?,size(a.head_dim)?,size(a.relative.dim)?,size(a.relative.extent)?);
    need(heads>0 && kv_heads>0 && heads.is_multiple_of(kv_heads) && head>0 && head.is_multiple_of(2) && relative>0 && extent>0 && epsilon>0.0,"the attention shape is not built")?;
    need(layout.kv_channels==kv_heads*head && layout.channels==width && layout.ring==a.window.is_some() && layout.kv_taps as u64==a.key_value_conv.kernel
        && layout.taps as u64==run.description.conv.kernel,"the state layout differs from the layer of the visit")?;
    let rows=step.total();
    need(!spans.is_empty() && states.len()==spans.len() && spans.iter().all(|span|span.count>0) && c.words(keep)==rows*width,"the rows of a step differ from the stream or from the chats")?;
    // Refused before any launch: a chat with more rows than its state takes in one step.
    need(spans.iter().all(|span|span.count<=room(layout,span.first)),"a step has more rows of one chat than its state takes")?;
    // Step 4. A layer with one attention norm gives the same weight under all three names.
    let norm=|x:&C::Value,count:usize,which:Weight|rounded_norm(c,x,count,width,epsilon,Some(w.words(which)?));
    let normed=match inputs {
        Inputs::Same(x) if !a.separate_inputs=>vec![norm(x,rows,Weight::QueryNorm(layer))?],
        Inputs::Same(x)=>vec![norm(x,rows,Weight::QueryNorm(layer))?,norm(x,rows,Weight::KeyNorm(layer))?,norm(x,rows,Weight::ValueNorm(layer))?],
        // Three inputs and one weight (a sliding layer that reads messages): one launch of all rows of the three. A
        // row is normed alone, so the words are those of three launches.
        Inputs::Each([q,k,v]) if !a.separate_inputs=>{
            let (all,part)=(norm(&c.join(&[q,k,v])?,3*rows,Weight::QueryNorm(layer))?,rows*width);
            vec![c.range(&all,0,part)?,c.range(&all,part,part)?,c.range(&all,2*part,part)?]
        },
        Inputs::Each([q,k,v])=>vec![norm(q,rows,Weight::QueryNorm(layer))?,norm(k,rows,Weight::KeyNorm(layer))?,norm(v,rows,Weight::ValueNorm(layer))?],
    };
    let (xq,xk,xv)=(&normed[0],&normed[normed.len()/2],&normed[normed.len()-1]);
    let product=|which:Weight,x:&C::Value,each:usize|->Result<C::Value> {
        let y=w.product_rows(c,which,x,rows)?;need(c.words(&y)==rows*each,"an attention product differs from the description")?;Ok(y)
    };
    // Step 5, the query side. The position factor is one word a row.
    let factor=match a.position_scale {
        Some(scale)=>Some(c.upload(&spans.iter().flat_map(|span|(span.first..).take(span.count).map(move |position|position_factor(scale,position))).collect::<Vec<_>>())?),
        None=>None,
    };
    let clip=|x:&C::Value,heads:usize,dim:usize,scale:&[f32],factor:Option<&C::Value>,threshold:f64|->Result<C::Value> {
        need(threshold>0.0 && c.words(x)==rows*heads*dim && (scale.is_empty() || scale.len()==heads*dim),"a clip input or scale differs from the description")?;
        let mut s=shape(Op::HeadClip);s.heads=number(heads)?;s.dim=number(dim)?;s.taps=u32::from(!scale.is_empty());s.cols=u32::from(factor.is_some());s.bound=threshold as f32;
        rows_operation(c,s,rows,[Some(x),factor,None,None],0,scale)
    };
    let query=clip(&product(Weight::Query(layer),xq,heads*head)?,heads,head,w.words(Weight::QueryClip(layer))?,factor.as_ref(),a.query_key_clip)?;
    let moved=clip(&product(Weight::Relative(layer),xq,heads*relative)?,heads,relative,w.words(Weight::RelativeClip(layer))?,factor.as_ref(),a.relative.clip)?;
    let table=w.words(Weight::RelativeTable(layer))?;need(table.len()==relative*extent,"the distance table differs from the description")?;
    let sink=product(Weight::Sink(layer),xq,heads)?;
    let gate=product(Weight::OutputGate(layer),xq,heads*head)?;
    // Step 5, the keys and the values of all rows. Each row reads a ring of its own; no ring of the state is written.
    let made=|which:Weight,x:&C::Value|->Result<Option<C::Value>> {
        match a.key_value_conv.taps {ConvTaps::Fixed {..}=>Ok(None),ConvTaps::FixedAndToken {..}=>product(which,x,kv_heads*layout.kv_taps).map(Some)}
    };
    let conv=|states:&mut [&mut VisitState<C::Value>],which:Ring,fixed:Weight,made:Option<&C::Value>,product:&C::Value|->Result<C::Value> {
        let (channels,taps)=layout.ring_size(which);let fixed=w.words(fixed)?;
        need(channels.is_multiple_of(kv_heads) && fixed.len()==channels*taps && made.is_none_or(|made|c.words(made)==rows*kv_heads*taps),"the taps of a key or value convolution differ from the description")?;
        let (rings,steps)=row_rings(c,layout,which,step,states,product)?;
        let mut s=shape(Op::ShortConv);s.heads=number(kv_heads)?;s.dim=number(channels/kv_heads)?;s.taps=number(taps)?;
        // The divisor counts each tap once, the made taps once more, and the input that is added back.
        s.cols=if made.is_some() {CONV_TOKEN_TAPS} else {0};s.bound=conv_scale(if made.is_some() {2*taps as u64+1} else {taps as u64+1});
        rows_operation(c,s,rows,[Some(&rings),made,None,Some(&steps)],0,fixed)
    };
    let key={
        let made=made(Weight::KeyTapProjection(layer),xk)?;
        let y=conv(states,Ring::Key,Weight::KeyTaps(layer),made.as_ref(),&product(Weight::Key(layer),xk,kv_heads*head)?)?;
        clip(&y,kv_heads,head,&[],None,a.query_key_clip)?
    };
    let value={
        let made=made(Weight::ValueTapProjection(layer),xv)?;
        let y=conv(states,Ring::Value,Weight::ValueTaps(layer),made.as_ref(),&product(Weight::Value(layer),xv,kv_heads*head)?)?;
        rounded_norm(c,&y,rows*kv_heads,head,epsilon,None)?
    };
    // The finished keys and values of all rows go into the planes of their chats before any row reads a plane.
    {
        let (keys,values,words)=(state::pack(c,&key)?,state::pack(c,&value)?,layout.row_words);let mut offset=0;
        for (span,state) in spans.iter().zip(states.iter_mut()) {
            write_rows(c,layout,&mut state.keys,*span,&c.range(&keys,offset*words,span.count*words)?)?;
            write_rows(c,layout,&mut state.values,*span,&c.range(&values,offset*words,span.count*words)?)?;
            offset+=span.count;
        }
    }
    // Step 5, the attention: once for each chat, in groups of rows. The views of the planes live only inside a chat's turn.
    let mut parts=Vec::new();let mut offset=0;
    for (span,state) in spans.iter().zip(states.iter()) {
        let (view,group)=(layout.view_rows(span.first+span.count as u64-1)?,heads/kv_heads);
        let (key_view,value_view)=(state::view(c,layout,&state.keys,view)?,state::view(c,layout,&state.values,view)?);
        // Bytes of one row over keys: the scores and the weights of each head over the view, and its relative term.
        let each=(key_bytes/4usize.saturating_mul(heads).saturating_mul(view.saturating_mul(2).saturating_add(extent)).max(1)).max(1);
        let mut done=0;
        while done<span.count {
            let (n,at)=(each.min(span.count-done),offset+done);
            let part=|x:&C::Value,words:usize|c.range(x,at*words,n*words);
            let (both,live)=key_steps(c,layout,span.first+done as u64,n)?;
            let mut s=shape(Op::RelBias);s.heads=number(heads)?;s.dim=number(relative)?;s.cols=number(extent)?;
            let bias=rows_operation(c,s,n,[Some(&part(&moved,heads*relative)?),None,None,None],0,table)?;
            let mut s=shape(Op::WindowScores);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(group)?;s.rows=number(view)?;s.cols=number(extent)?;s.bound=score_scale(a.head_dim);
            let scores=rows_operation(c,s,n,[Some(&part(&query,heads*head)?),Some(&key_view),Some(&bias),Some(&both)],shared(1),&[])?;
            let mut s=shape(Op::SinkSoftmax);s.heads=number(heads)?;s.rows=number(view)?;
            let weights=rows_operation(c,s,n,[Some(&scores),Some(&part(&sink,heads)?),Some(&live),None],0,&[])?;
            let mut s=shape(Op::WindowValues);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(group)?;s.rows=number(view)?;
            parts.push(rows_operation(c,s,n,[Some(&weights),Some(&value_view),Some(&part(&gate,heads*head)?),Some(&both)],shared(1),&[])?);
            done+=n;
        }
        offset+=span.count;
    }
    let attended=if parts.len()==1 {parts.remove(0)} else {c.join(&parts.iter().collect::<Vec<_>>())?};
    need(c.words(&attended)==rows*heads*head,"the attention output differs from the description")?;
    // Steps 6 and 7. The output product is the new input of the ring after attention; one launch gives the convolution
    // and the first sum.
    let output=product(Weight::Output(layer),&attended,width)?;
    let (rings,steps)=row_rings(c,layout,Ring::AfterAttention,step,states,&output)?;
    let taps=w.words(Weight::ConvAfterAttention(layer))?;need(taps.len()==width*layout.taps,"the taps after attention differ from the description")?;
    let mut s=shape(Op::ShortConv);s.heads=1;s.dim=number(width)?;s.taps=number(layout.taps)?;s.bound=conv_scale(layout.taps as u64+1);
    if spec.boundary {return rows_operation(c,s,rows,[Some(&rings),None,None,Some(&steps)],0,taps);}
    s.cols=CONV_ADDEND;rows_operation(c,s,rows,[Some(&rings),None,Some(keep),Some(&steps)],0,taps)
}
#[cfg(test)]
mod tests;
