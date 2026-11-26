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
//! - A decode cohort passes each chat's KV plane in the scores and values commands. Rows of one
//!   prompt share its planes. Both forms use each row's own live count and slot.
//!
//! The part over keys (the relative term, the scores, the softmax, the weighted values) runs in groups of rows of
//! one chat, so that the scores of one group stay under the byte limit of the layout (`Layout::key_bytes`). A group changes no word: no row reads a score
//! of another row.
//!
//! No ring of the state is written here. The rings after the step are pending in the state until
//! `state::commit`. The planes are written: only slots of the positions of this step, which no row of a
//! position before the step can see.
use super::{position_factor,rounded_norm,score_scale};
use super::super::{experts::conv_scale,mix::{Inputs,Run,Weight,Weights,need,number,shape},rows::{Rows,gather,rows_operation,shared},state::{self,Layout,Ring,VisitState,size,many::{convolve,convolve_kv,direct_parts,held_apart,key_steps,key_steps_both,room,write_rows}}};
use crate::{Result,bounded_decoder::{Compute,KvFinishPair,WindowOne,WindowSplit}};
use engine_model::plan::description::{ConvTaps,Visit};
use spark_gpu_probe::{block_stream::{CONV_ADDEND,CONV_TOKEN_TAPS},causal_ops::Op};

/// Steps 4 to 7 of one visit for the rows of a step. `states` holds the state of this visit of each chat, in step
/// order. `keep` is the stream that the visit read, `[rows][width]`. `inputs` are the inputs of the attention
/// norms, each `[rows][width]`. The result is the residual inside the block, `[rows][width]`, 32-bit and not
/// rounded. With one chat and one row it gives the words of `half`: a row has the words of `half` at its position
/// after the rows before it in its chat.
pub fn half_rows<C:Compute,W:Weights<C>>(run:&Run<'_,C,W>,visit:&Visit,layout:&Layout,step:&Rows,states:&mut [&mut VisitState<C::Value>],
    keep:&C::Value,inputs:&Inputs<C::Value>,prefill:bool)->Result<C::Value> {
    let (spans,key_bytes)=(step.chats(),layout.key_bytes);
    let (layer,spec)=run.layer(visit)?;let (c,w,a,width,epsilon)=(run.compute,run.weights,spec.attention,run.width(),run.description.epsilon);
    let (heads,kv_heads,head,relative,extent)=(size(a.query_heads)?,size(a.kv_heads)?,size(a.head_dim)?,size(a.relative.dim)?,size(a.relative.extent)?);
    need(heads>0 && kv_heads>0 && heads.is_multiple_of(kv_heads) && head>0 && head.is_multiple_of(2) && relative>0 && extent>0 && epsilon>0.0,"the attention shape is not built")?;
    need(layout.kv_channels==kv_heads*head && layout.channels==width && layout.ring==a.window.is_some() && layout.kv_taps as u64==a.key_value_conv.kernel
        && layout.taps as u64==run.description.conv.kernel,"the state layout differs from the layer of the visit")?;
    let rows=step.total();
    need(!spans.is_empty() && states.len()==spans.len() && spans.iter().all(|span|span.count>0) && c.words(keep)==rows*width,"the rows of a step differ from the stream or from the chats")?;
    // Refused before any launch: a chat with more rows than its state takes in one step.
    // A chat with more than one row in the step works in wide planes of a window layer (`Layout::wide`), made from its
    // state before any write; the state keeps its planes until the step is done.
    // The wide ring of a chat is `window + count - 1` rows for its `count` rows in this step, at most the ring of the
    // step row limit: the order of a sum over keys is by distance, so the ring size changes no word, and a check of a
    // few drafted rows does not clear and copy a ring of the whole limit.
    let fits=spans.iter().map(|span|Layout {step_rows:span.count.min(layout.step_rows),..*layout}).collect::<Vec<_>>();
    need(spans.iter().zip(&fits).all(|(span,fit)|span.count<=room(&fit.wide(),span.first)),"a step has more rows of one chat than its state takes")?;
    // Step 4. A layer with one attention norm gives the same weight under all three names.
    let norm=|x:&C::Value,count:usize,which:Weight|rounded_norm(c,x,count,width,epsilon,Some(w.words(which)?));
    let (normed,inputs_half)=match inputs {
        Inputs::NormedHalfSame(x)=>(vec![c.range(x,0,c.words(x))?],true),
        Inputs::NormedHalfEach([q,k,v])=>(vec![c.range(q,0,c.words(q))?,c.range(k,0,c.words(k))?,c.range(v,0,c.words(v))?],true),
        Inputs::NormedSame(x)=>(vec![c.range(x,0,c.words(x))?],false),
        Inputs::NormedEach([q,k,v])=>(vec![c.range(q,0,c.words(q))?,c.range(k,0,c.words(k))?,c.range(v,0,c.words(v))?],false),
        Inputs::Same(x) if !a.separate_inputs=>(vec![norm(x,rows,Weight::QueryNorm(layer))?],false),
        Inputs::Same(x)=>(vec![norm(x,rows,Weight::QueryNorm(layer))?,norm(x,rows,Weight::KeyNorm(layer))?,norm(x,rows,Weight::ValueNorm(layer))?],false),
        // A prompt step of such a layer: three launches on the three inputs as they lie, with no join copy of
        // `3 * rows * width` words. A row is normed alone, so the words are those of the one launch below.
        Inputs::Each([q,k,v]) if !a.separate_inputs && prefill=>{
            let weight=Weight::QueryNorm(layer);
            (vec![norm(q,rows,weight)?,norm(k,rows,weight)?,norm(v,rows,weight)?],false)
        },
        // Three inputs and one weight (a sliding layer that reads messages): one launch of all rows of the three. A
        // row is normed alone, so the words are those of three launches.
        Inputs::Each([q,k,v]) if !a.separate_inputs=>{
            let (all,part)=(norm(&c.join(&[q,k,v])?,3*rows,Weight::QueryNorm(layer))?,rows*width);
            (vec![c.range(&all,0,part)?,c.range(&all,part,part)?,c.range(&all,2*part,part)?],false)
        },
        Inputs::Each([q,k,v])=>(vec![norm(q,rows,Weight::QueryNorm(layer))?,norm(k,rows,Weight::KeyNorm(layer))?,norm(v,rows,Weight::ValueNorm(layer))?],false),
    };
    let (xq,xk,xv)=(&normed[0],&normed[normed.len()/2],&normed[normed.len()-1]);
    let product=|which:Weight,x:&C::Value,each:usize|->Result<C::Value> {
        let y=if inputs_half {
            w.product_rows_half_in(c,which,x,rows)?.ok_or_else(||"loop decoder: half-input attention product unavailable".to_owned())?
        } else {
            w.product_rows(c,which,x,rows)?
        };
        need(c.words(&y)==rows*each,"an attention product differs from the description")?;Ok(y)
    };
    // Step 5, the query side. The position factor is one word a row.
    let factor=match a.position_scale {
        Some(scale)=>Some(c.upload_constant(&spans.iter().flat_map(|span|(span.first..).take(span.count).map(move |position|position_factor(scale,position))).collect::<Vec<_>>())?),
        None=>None,
    };
    let clip=|x:&C::Value,heads:usize,dim:usize,scale:&[f32],factor:Option<&C::Value>,threshold:f64|->Result<C::Value> {
        need(threshold>0.0 && c.words(x)==rows*heads*dim && (scale.is_empty() || scale.len()==heads*dim),"a clip input or scale differs from the description")?;
        let mut s=shape(Op::HeadClip);s.heads=number(heads)?;s.dim=number(dim)?;s.taps=u32::from(!scale.is_empty());s.cols=u32::from(factor.is_some());s.bound=threshold as f32;
        rows_operation(c,s,rows,[Some(x),factor,None,None],0,scale)
    };
    // The products on one input go in the stacks of the token step (`attention::half`): where the owner holds the
    // matrices one after another, one product of rows over the stack (each row's products side by side), then one
    // gather puts the rows of each matrix together. A product word is one chain over its own row, so the words are
    // those of one product a matrix. A provider without a contiguous stack runs one product per matrix.
    let stacked=|names:&[Weight],sizes:&[usize],x:&C::Value|->Result<Vec<C::Value>> {
        let total=sizes.iter().sum::<usize>();
        let separate=||names.iter().zip(sizes).map(|(&name,&size)|product(name,x,size)).collect::<Result<Vec<_>>>();
        if inputs_half {
            if let Some(parts)=w.stack_products_rows_half_in(c,names,x,rows)? {
                need(parts.len()==sizes.len() && parts.iter().zip(sizes).all(|(y,&size)|c.words(y)==rows*size),"an attention product differs from the description")?;
                return Ok(parts);
            }
            return separate();
        }
        if let Some(parts)=w.stack_products_rows(c,names,x,rows)? {
            need(parts.len()==sizes.len() && parts.iter().zip(sizes).all(|(y,&size)|c.words(y)==rows*size),"an attention product differs from the description")?;
            return Ok(parts);
        }
        let Some(y)=w.stack_product_rows(c,names,x,rows)? else {return separate()};
        need(total>0 && c.words(&y)==rows*total,"an attention product differs from the description")?;
        // One row: the parts already lie one after another. More rows: the gather moves units of the largest width
        // that divides every part, matrix after matrix, row after row.
        let parts=if rows==1 {y} else {
            let unit=sizes.iter().fold(0,|a,&b|common(a,b));
            let mut table=Vec::with_capacity(rows*total/unit);let mut offset=0;
            for &size in sizes {for row in 0..rows {table.extend((0..size/unit).map(|j|(row*total+offset)/unit+j));}offset+=size;}
            // The stacked rows go as soon as the gather has them.
            let parts=gather(c,&y,&table,unit)?;drop(y);parts
        };
        let mut start=0;
        sizes.iter().map(|&size|{let part=c.range(&parts,start,rows*size);start+=rows*size;part}).collect()
    };
    // With one input for all three, the key and the value are in the stack of the query input too.
    let one=normed.len()==1;
    let names=[Weight::Query(layer),Weight::Relative(layer),Weight::Sink(layer),Weight::OutputGate(layer),Weight::Key(layer),Weight::Value(layer)];
    let sizes=[heads*head,heads*relative,heads,heads*head,kv_heads*head,kv_heads*head];
    let count=if one {6} else {4};
    let mut on_query=stacked(&names[..count],&sizes[..count],xq)?.into_iter();
    let mut next=||on_query.next().ok_or("loop decoder: an attention product is missing".to_owned());
    // The query and the relative state as products: their clips run where the arms read them (`clipped`), or inside
    // the one-launch attention of a check pass (`Compute::window_one`), which gives the same words.
    let (products,sink,gate)=((next()?,next()?),next()?,next()?);
    let (query_scale,moved_scale)=(w.words(Weight::QueryClip(layer))?,w.words(Weight::RelativeClip(layer))?);
    let mut clips:Option<(C::Value,C::Value)>=None;
    let mut clipped=||->Result<(C::Value,C::Value)> {
        if clips.is_none() {clips=Some((clip(&products.0,heads,head,query_scale,factor.as_ref(),a.query_key_clip)?,clip(&products.1,heads,relative,moved_scale,factor.as_ref(),a.relative.clip)?));}
        let (query,moved)=clips.as_ref().ok_or("loop decoder: the clipped query is missing")?;
        Ok((c.range(query,0,c.words(query))?,c.range(moved,0,c.words(moved))?))
    };
    let table=w.words(Weight::RelativeTable(layer))?;need(table.len()==relative*extent,"the distance table differs from the description")?;
    // Step 5, the keys and the values of all rows. Each row reads a ring of its own; no ring of the state is written.
    let made=|which:Weight,x:&C::Value|->Result<Option<C::Value>> {
        match a.key_value_conv.taps {ConvTaps::Fixed {..}=>Ok(None),ConvTaps::FixedAndToken {..}=>product(which,x,kv_heads*layout.kv_taps).map(Some)}
    };
    // The key or the value product, and in a layer that makes taps from the token the product of those taps on the
    // same input: one stack.
    let with_taps=|main:Weight,taps:Weight,x:&C::Value|->Result<(C::Value,Option<C::Value>)> {
        if matches!(a.key_value_conv.taps,ConvTaps::Fixed {..}) {return Ok((product(main,x,kv_heads*head)?,None));}
        let mut y=stacked(&[main,taps],&[kv_heads*head,kv_heads*layout.kv_taps],x)?.into_iter();
        Ok((y.next().ok_or("loop decoder: a key or value product is missing")?,Some(y.next().ok_or("loop decoder: a tap product is missing")?)))
    };
    let ((key_product,key_made),(value_product,value_made))=if one {
        ((next()?,made(Weight::KeyTapProjection(layer),xk)?),(next()?,made(Weight::ValueTapProjection(layer),xv)?))
    } else {(with_taps(Weight::Key(layer),Weight::KeyTapProjection(layer),xk)?,with_taps(Weight::Value(layer),Weight::ValueTapProjection(layer),xv)?)};
    let convolution = |which: Ring, fixed: Weight, made: Option<&C::Value>| -> Result<_> {
        let (channels,taps)=layout.ring_size(which);let fixed=w.words(fixed)?;
        need(channels.is_multiple_of(kv_heads) && fixed.len()==channels*taps && made.is_none_or(|made|c.words(made)==rows*kv_heads*taps),"the taps of a key or value convolution differ from the description")?;
        let mut s=shape(Op::ShortConv);s.heads=number(kv_heads)?;s.dim=number(channels/kv_heads)?;s.taps=number(taps)?;
        // The divisor counts each tap once, the made taps once more, and the input that is added back.
        s.cols=if made.is_some() {CONV_TOKEN_TAPS} else {0};s.bound=conv_scale(if made.is_some() {2*taps as u64+1} else {taps as u64+1});
        Ok((s, fixed))
    };
    let (key_conv, key_fixed) = convolution(Ring::Key, Weight::KeyTaps(layer), key_made.as_ref())?;
    let (value_conv, value_fixed) = convolution(Ring::Value, Weight::ValueTaps(layer), value_made.as_ref())?;
    // Paired multi-row KV finish for kept decode passes and multi-chat cohorts (`head == 128`): fuses Key and Value
    // `ShortConv`, `HeadClip`/`RoundedNorm`, `PackBf16`, and ring capture into ceil(rows / 16) batched launches.
    let mut fused_kv: Option<(C::Value, C::Value)> = None;
    let mut direct_kv = false;
    let separate = rows == spans.len() && spans.iter().all(|span| span.count == 1);
    if !prefill && rows > 1 && head == 128 && ((step.keeps() && spans.len() == 1) || separate) {
        let ((k_ch, k_taps), (v_ch, v_taps)) = (layout.ring_size(Ring::Key), layout.ring_size(Ring::Value));
        if k_taps >= 2 && k_taps == v_taps && k_ch == kv_heads * head && v_ch == kv_heads * head {
            let (k_fixed, v_fixed) = (w.words(Weight::KeyTaps(layer))?, w.words(Weight::ValueTaps(layer))?);
            need(k_fixed.len() == k_ch * k_taps && v_fixed.len() == v_ch * v_taps
                && key_made.as_ref().is_none_or(|m| c.words(m) == rows * kv_heads * k_taps)
                && value_made.as_ref().is_none_or(|m| c.words(m) == rows * kv_heads * v_taps),
                "the taps of a key or value convolution differ from the description")?;
            let k_scale = conv_scale(if key_made.is_some() { 2 * k_taps as u64 + 1 } else { k_taps as u64 + 1 });
            let v_scale = conv_scale(if value_made.is_some() { 2 * v_taps as u64 + 1 } else { v_taps as u64 + 1 });
            let ring_words = k_taps * k_ch;
            // In launches of at most `DIRECT_CHATS_MAX` chats (`direct_parts`). A row reads its own rings and writes its
            // own plane row, so its words do not depend on the other rows of its launch.
            if separate && !step.keeps() && k_taps <= spark_gpu_probe::fast_ops::DIRECT_TAPS_MAX {
                let plane_slots = spans.iter().map(|s| layout.slot(s.first)).collect::<Result<Vec<_>>>()?;
                let parts = direct_parts(rows);
                let mut made = Vec::with_capacity(parts.len());
                for &(start, n) in &parts {
                    // One part takes the products as they are. More parts take views of their rows.
                    let part = |x: &C::Value, words: usize| (n < rows).then(|| c.range(x, start * words, n * words)).transpose();
                    let (k_in, v_in) = (part(&key_product, k_ch)?, part(&value_product, v_ch)?);
                    let k_made = key_made.as_ref().map(|m| part(m, kv_heads * k_taps)).transpose()?.flatten();
                    let v_made = value_made.as_ref().map(|m| part(m, kv_heads * v_taps)).transpose()?.flatten();
                    let req = crate::bounded_decoder::KvFinishDirect {
                        rows: n, kv_heads, taps: k_taps, plane_rows: layout.rows,
                        key_bound: a.query_key_clip as f32, val_eps: epsilon as f32, key_scale: k_scale, val_scale: v_scale,
                        key_in: k_in.as_ref().unwrap_or(&key_product), key_fixed: k_fixed, key_made: k_made.as_ref().or(key_made.as_ref()),
                        val_in: v_in.as_ref().unwrap_or(&value_product), val_fixed: v_fixed, val_made: v_made.as_ref().or(value_made.as_ref()),
                    };
                    let mut targets: Vec<_> = states[start..start + n].iter_mut().zip(&spans[start..start + n]).zip(&plane_slots[start..start + n])
                        .map(|((st, sp), &ps)| (&st.rings[Ring::Key as usize], &st.rings[Ring::Value as usize], (sp.first % k_taps as u64) as usize, &mut st.keys, &mut st.values, ps))
                        .collect();
                    let Some([ke, ve]) = c.kv_finish_direct(&req, &mut targets)? else { break };
                    drop(targets);
                    need(c.words(&ke) == n * ring_words && c.words(&ve) == n * ring_words,
                        "the direct key-value finish differs from the layout")?;
                    made.push((ke, ve));
                }
                // A refused part leaves the whole step to the forms below. They write the plane rows of every chat
                // again, with the same words, before any row reads them; no pending ring is set before every part ran.
                if made.len() == parts.len() {
                    for (&(start, n), (ke, ve)) in parts.iter().zip(&made) {
                        for (row, state) in states[start..start + n].iter_mut().enumerate() {
                            state.next[Ring::Key as usize] = Some(c.range(ke, row * ring_words, ring_words)?);
                            state.next[Ring::Value as usize] = Some(c.range(ve, row * ring_words, ring_words)?);
                        }
                    }
                    direct_kv = true;
                }
            }
            if !direct_kv && (!separate || rows <= 16) {
                let first_slot = (spans[0].first % k_taps as u64) as usize;
                let chat_rings = if separate {
                    states.iter().zip(spans).map(|(state, span)| (&state.rings[Ring::Key as usize],
                        &state.rings[Ring::Value as usize], (span.first % k_taps as u64) as usize)).collect()
                } else { Vec::new() };
                let state = &states[0];
                let pair = KvFinishPair {
                    rows, kv_heads, taps: k_taps, first_slot, chat_rings: &chat_rings,
                    key_bound: a.query_key_clip as f32, val_eps: epsilon as f32, key_scale: k_scale, val_scale: v_scale,
                    key_in: &key_product, key_ring: &state.rings[Ring::Key as usize], key_fixed: k_fixed, key_made: key_made.as_ref(),
                    val_in: &value_product, val_ring: &state.rings[Ring::Value as usize], val_fixed: v_fixed, val_made: value_made.as_ref(),
                };
                if let Some([kp, ke, vp, ve]) = c.kv_finish_pair(&pair)? {
                    need(c.words(&kp) == rows * layout.row_words && c.words(&vp) == rows * layout.row_words
                        && c.words(&ke) == rows * ring_words && c.words(&ve) == rows * ring_words,
                        "the paired key-value finish differs from the layout")?;
                    if separate {
                        for (row, state) in states.iter_mut().enumerate() {
                            state.next[Ring::Key as usize] = Some(c.range(&ke, row * ring_words, ring_words)?);
                            state.next[Ring::Value as usize] = Some(c.range(&ve, row * ring_words, ring_words)?);
                        }
                    } else {
                        let state = &mut states[0];
                        state.next[Ring::Key as usize] = Some(c.range(&ke, (rows - 1) * ring_words, ring_words)?);
                        state.every[Ring::Key as usize] = Some(ke);
                        state.next[Ring::Value as usize] = Some(c.range(&ve, (rows - 1) * ring_words, ring_words)?);
                        state.every[Ring::Value as usize] = Some(ve);
                    }
                    fused_kv = Some((kp, vp));
                }
            } else if !direct_kv {
                let mut chunks = Vec::with_capacity(rows.div_ceil(16));
                let mut done = 0usize;
                for start in (0..rows).step_by(16) {
                    let n = 16.min(rows - start);
                    let first_slot = (spans[start].first % k_taps as u64) as usize;
                    let chat_rings: Vec<_> = states[start..start + n].iter().zip(&spans[start..start + n])
                        .map(|(state, span)| (&state.rings[Ring::Key as usize], &state.rings[Ring::Value as usize], (span.first % k_taps as u64) as usize))
                        .collect();
                    let k_in = c.range(&key_product, start * k_ch, n * k_ch)?;
                    let v_in = c.range(&value_product, start * v_ch, n * v_ch)?;
                    let k_made = key_made.as_ref().map(|m| c.range(m, start * kv_heads * k_taps, n * kv_heads * k_taps)).transpose()?;
                    let v_made = value_made.as_ref().map(|m| c.range(m, start * kv_heads * v_taps, n * kv_heads * v_taps)).transpose()?;
                    let state = &states[start];
                    let pair = KvFinishPair {
                        rows: n, kv_heads, taps: k_taps, first_slot, chat_rings: &chat_rings,
                        key_bound: a.query_key_clip as f32, val_eps: epsilon as f32, key_scale: k_scale, val_scale: v_scale,
                        key_in: &k_in, key_ring: &state.rings[Ring::Key as usize], key_fixed: k_fixed, key_made: k_made.as_ref(),
                        val_in: &v_in, val_ring: &state.rings[Ring::Value as usize], val_fixed: v_fixed, val_made: v_made.as_ref(),
                    };
                    let Some([kp, ke, vp, ve]) = c.kv_finish_pair(&pair)? else { break };
                    need(c.words(&kp) == n * layout.row_words && c.words(&vp) == n * layout.row_words
                        && c.words(&ke) == n * ring_words && c.words(&ve) == n * ring_words,
                        "the paired key-value finish differs from the layout")?;
                    chunks.push((start, n, kp, ke, vp, ve));
                    done += n;
                }
                if done == rows {
                    for (start, n, _, ke, _, ve) in &chunks {
                        for row in 0..*n {
                            states[start + row].next[Ring::Key as usize] = Some(c.range(ke, row * ring_words, ring_words)?);
                            states[start + row].next[Ring::Value as usize] = Some(c.range(ve, row * ring_words, ring_words)?);
                        }
                    }
                    let kps: Vec<_> = chunks.iter().map(|(_, _, kp, _, _, _)| kp).collect();
                    let vps: Vec<_> = chunks.iter().map(|(_, _, _, _, vp, _)| vp).collect();
                    fused_kv = Some((c.join(&kps)?, c.join(&vps)?));
                }
            }
        }
    }
    // The wide planes of each chat with more than one row in the step of a window layer whose ring is wider for a step.
    // A check pass of many chats writes the new rows of a chat into its ring where the ring takes them (`held_apart`):
    // before the first wrap they take slots that were never written, and past it the spare rows of the ring
    // (`state::spare_rows`) keep them off every key that a row from the chat's position on reads. A retry writes those
    // slots again before it reads them, also after the chat kept only some of the rows.
    let wraps:Vec<bool>=spans.iter().map(|&span|held_apart(layout,step,span)).collect();
    let any_wraps=wraps.iter().any(|&w|w);
    let mut scratch=spans.iter().zip(states.iter()).zip(&fits).map(|((span,state),fit)|if !step.keeps() && span.count>1 && fit.wide().rows>layout.rows && span.first+span.count as u64>layout.rows as u64 {
        Ok(Some([state::many::widen(c,fit,&state.keys,span.first)?,state::many::widen(c,fit,&state.values,span.first)?]))} else {Ok(None)}).collect::<Result<Vec<_>>>()?;
    // Prompts (single or multi-chat) with writable KV planes fuse ShortConv + Clip/Norm + PackBf16 + plane write per chat.
    let fused = if prefill && !step.keeps() && spans.len() == states.len()
        && spans.iter().all(|s| s.count >= key_conv.taps as usize && s.count >= value_conv.taps as usize && s.count >= 2)
    {
        let (kv_ch, k_taps, v_taps) = (kv_heads * head, key_conv.taps as usize, value_conv.taps as usize);
        let (mut ok, mut offset) = (true, 0usize);
        for (((span, state), scratch), fit) in spans.iter().zip(states.iter_mut()).zip(scratch.iter_mut()).zip(fits.iter()) {
            let plane_layout = if scratch.is_some() { fit.wide() } else { *layout };
            let first = plane_layout.slot(span.first)?;
            let plane_rows = if scratch.is_some() || plane_layout.ring {
                plane_layout.rows
            } else {
                state::plane_rows(c, &plane_layout, &state.keys)?
            };
            let write = |clip, param| spark_gpu_probe::fast_ops::KvWrite { first, plane_rows, clip, param };
            let (k_slice, v_slice) = if spans.len() == 1 {
                (None, None)
            } else {
                (Some(c.range(&key_product, offset * kv_ch, span.count * kv_ch)?),
                 Some(c.range(&value_product, offset * kv_ch, span.count * kv_ch)?))
            };
            let k_m = if spans.len() == 1 { None } else { key_made.as_ref().map(|m| c.range(m, offset * kv_heads * k_taps, span.count * kv_heads * k_taps)).transpose()? };
            let v_m = if spans.len() == 1 { None } else { value_made.as_ref().map(|m| c.range(m, offset * kv_heads * v_taps, span.count * kv_heads * v_taps)).transpose()? };
            let kp = k_slice.as_ref().unwrap_or(&key_product);
            let vp = v_slice.as_ref().unwrap_or(&value_product);
            let km = if spans.len() == 1 { key_made.as_ref() } else { k_m.as_ref() };
            let vm = if spans.len() == 1 { value_made.as_ref() } else { v_m.as_ref() };
            let key = convolve_kv(c, Ring::Key, *span, key_conv, kp, km, key_fixed,
                state, scratch.as_mut().map(|planes| &mut planes[0]), write(true, a.query_key_clip as f32))?;
            if !key || !convolve_kv(c, Ring::Value, *span, value_conv, vp, vm, value_fixed,
                state, scratch.as_mut().map(|planes| &mut planes[1]), write(false, epsilon as f32))?
            {
                ok = false; break;
            }
            offset += span.count;
        }
        ok
    } else {
        false
    };
    // If a plane declined the fused write, the original path rewrites the same rows before attention.
    if !fused && !direct_kv {
        let (keys,values) = if let Some(pair) = fused_kv {
            pair
        } else {
            let key = convolve(c,layout,Ring::Key,step,states,key_conv,&key_product,key_made.as_ref(),None,key_fixed)?;
            let key = clip(&key,kv_heads,head,&[],None,a.query_key_clip)?;
            let value = convolve(c,layout,Ring::Value,step,states,value_conv,&value_product,value_made.as_ref(),None,value_fixed)?;
            let value = rounded_norm(c,&value,rows*kv_heads,head,epsilon,None)?;
            (state::pack(c,&key)?,state::pack(c,&value)?)
        };
        let words=layout.row_words;let mut offset=0;
        for ((((span,state),scratch),fit),&wrap) in spans.iter().zip(states.iter_mut()).zip(scratch.iter_mut()).zip(&fits).zip(&wraps) {
            let (keys,values)=(c.range(&keys,offset*words,span.count*words)?,c.range(&values,offset*words,span.count*words)?);
            // Keep the verify writes of a chat that is held apart out of `state.keys`/`state.values` until `accept()`
            // commits the kept prefix. Straight planes and rings that take the rows (`count <= room`) write directly:
            // into unused future slots (`>= span.first`) before the first wrap, and past it over rows that are out of
            // the window of every position from `span.first` on. Every key that the chat still sees stays untouched on
            // partial accept or discard, and multi-chat verify shares the batched `window_split` path with zero
            // `accept()` KV copies.
            if wrap {state.written=Some([keys,values]);}
            else if let Some([wide_keys,wide_values])=scratch {
                let wide=fit.wide();write_rows(c,&wide,wide_keys,*span,&keys)?;write_rows(c,&wide,wide_values,*span,&values)?;
            }
            else {write_rows(c,layout,&mut state.keys,*span,&keys)?;write_rows(c,layout,&mut state.values,*span,&values)?;}
            offset+=span.count;
        }
    }
    // Step 5. Independent decode rows share a launch. A prompt splits into groups within its chat.
    let mut parts=Vec::new();let mut offset=0;let mut attended_half=false;
    // Before the first wrap, all live keys lie in the prefix. After a wrap a
    // ring uses its full extent. A common view lets a decode cohort
    // run all four attention operations once, retaining the per-chat distance order.
    let last=spans.iter().map(|span|span.first+span.count as u64-1).max().unwrap_or(0);
    let short=usize::try_from(last.saturating_add(1)).unwrap_or(usize::MAX).min(layout.rows);
    let view=short.checked_next_multiple_of(32).unwrap_or(layout.rows).min(layout.rows);
    let words=rows.saturating_mul(view).saturating_mul(2*heads)
        .saturating_add(rows.saturating_mul(heads).saturating_mul(extent));
    // Multi-chat decode & unwrapped multi-chat verify (`C >= 2`): run GQA-packed Split-KV in chunks of
    // `FW_SPLIT_ROWS_MAX` (16), bit-exact with C1 at any context length and avoiding per-chat attention launches.
    let mut cohort_done = false;
    let split_max = spark_gpu_probe::flash_window::FW_SPLIT_ROWS_MAX;
    if !prefill && !any_wraps && spans.len() >= 2 && scratch.iter().all(Option::is_none) {
        let mut row_meta: Vec<(usize, u64)> = Vec::with_capacity(rows);
        for (chat, span) in spans.iter().enumerate() {
            for r in 0..span.count {
                row_meta.push((chat, span.first + r as u64));
            }
        }
        let mut cohort_parts = Vec::with_capacity(rows.div_ceil(split_max));
        let mut done = 0usize;
        for start in (0..rows).step_by(split_max) {
            let n = split_max.min(rows - start);
            let mut steps_v = Vec::with_capacity(2 * n);
            let mut row_pos_live = Vec::with_capacity(n);
            let mut planes = Vec::with_capacity(n);
            for &(chat, pos) in &row_meta[start..start + n] {
                let live = layout.live(pos)?;
                steps_v.extend([live, layout.slot(pos)?]);
                row_pos_live.push((pos, live));
                let state = &states[chat];
                let chat_view = if layout.ring {
                    view
                } else {
                    layout.view_rows_in(pos, state::plane_rows(c, layout, &state.keys)?)?
                };
                planes.push((state::view(c, layout, &state.keys, chat_view)?, state::view(c, layout, &state.values, chat_view)?));
            }
            let both = state::step_words(c, &steps_v)?;
            let (q_part, m_part, s_part, g_part, pos_part) = if n == rows {
                (None, None, None, None, None)
            } else {
                let part = |x: &C::Value, words: usize| c.range(x, start * words, n * words);
                (Some(part(&products.0, heads * head)?), Some(part(&products.1, heads * relative)?),
                    Some(part(&sink, heads)?), Some(part(&gate, heads * head)?),
                    match &factor { Some(f) => Some(part(f, 1)?), None => None })
            };
            let split_req = WindowSplit {
                base: WindowOne {
                    rows: n, heads, kv_heads, view, extent, scale: score_scale(a.head_dim),
                    query: q_part.as_ref().unwrap_or(&products.0), query_scale, query_bound: a.query_key_clip as f32,
                    moved: m_part.as_ref().unwrap_or(&products.1), moved_scale, moved_bound: a.relative.clip as f32,
                    position: if n == rows { factor.as_ref() } else { pos_part.as_ref() },
                    keys: &planes[0].0, values: &planes[0].1, written: None, rel_table: table,
                    sink: s_part.as_ref().unwrap_or(&sink), gate: g_part.as_ref().unwrap_or(&gate), steps: &both,
                },
                is_ring: layout.ring, first_pos: 0, row_pos_live: &row_pos_live, delta: None, planes: &planes,
            };
            let Some(output) = c.window_split(&split_req)? else { break };
            need(c.words(&output) == n * heads * head, "the split-kv cohort attention differs from the description")?;
            cohort_parts.push(output);
            done += n;
        }
        if done == rows {
            parts.extend(cohort_parts);
            cohort_done = true;
        }
    }
    // Multi-chat prefill packed-BF16 attention (`window_rows_half`): lets `Weight::Output` run `HALF_IN` across all chats.
    if !cohort_done && prefill && !any_wraps && spans.len() >= 2 && w.prefers_half_output(layer)
        && spans.iter().all(|s| s.count >= spark_gpu_probe::flash_window::FW_ROWS_MIN)
    {
        let (mut half_parts, mut chat_off, mut ok) = (Vec::with_capacity(spans.len()), 0usize, true);
        for (((span, state), scratch), fit) in spans.iter().zip(states.iter()).zip(&scratch).zip(&fits) {
            let wide = fit.wide();
            let (chat_layout, planes) = match scratch { Some([k, v]) => (&wide, [k, v]), None => (layout, [&state.keys, &state.values]) };
            let chat_view = chat_layout.view_rows_in(span.first + span.count as u64 - 1, state::plane_rows(c, chat_layout, planes[0])?)?;
            let (kv, vv) = (state::view(c, chat_layout, planes[0], chat_view)?, state::view(c, chat_layout, planes[1], chat_view)?);
            let mut done = 0usize;
            while done < span.count {
                let (n, at) = (spark_gpu_probe::flash_window::FW_ROWS_MAX.min(span.count - done), chat_off + done);
                let both = key_steps_both(c, chat_layout, span.first + done as u64, n)?;
                let part = |x: &C::Value, words: usize| c.range(x, at * words, n * words);
                let position = match &factor { Some(f) => Some(part(f, 1)?), None => None };
                let one = WindowOne {
                    rows: n, heads, kv_heads, view: chat_view, extent, scale: score_scale(a.head_dim),
                    query: &part(&products.0, heads * head)?, query_scale, query_bound: a.query_key_clip as f32,
                    moved: &part(&products.1, heads * relative)?, moved_scale, moved_bound: a.relative.clip as f32,
                    position: position.as_ref(), keys: &kv, values: &vv, written: None, rel_table: table,
                    sink: &part(&sink, heads)?, gate: &part(&gate, heads * head)?, steps: &both,
                };
                match w.window_rows_half(c, &one, span.first + done as u64)? {
                    Some(out) => { need(c.words(&out) == n * heads * head / 2, "the half prompt attention differs from the description")?; half_parts.push(out); done += n; }
                    None => { ok = false; break; }
                }
            }
            if !ok { break; }
            chat_off += span.count;
        }
        if ok {
            parts.extend(half_parts);
            attended_half = true;
            cohort_done = true;
        }
    }
    // A verify pass (provisional rows) keeps its own chronological views below.
    if cohort_done {
    } else if !prefill && !any_wraps && (2..=16).contains(&rows) && rows==spans.len() && words<=key_bytes/4
        && states.iter().all(|s|c.words(&s.keys)>=view*layout.row_words && c.words(&s.values)>=view*layout.row_words) {
        let mut steps=Vec::with_capacity(2*rows);let mut counts=Vec::with_capacity(rows);
        for span in spans {let n=layout.live(span.first)?;steps.extend([n,layout.slot(span.first)?]);counts.push(n);}
        let both=state::step_words(c,&steps)?;
        let planes=states.iter().map(|state|Ok((state::view(c,layout,&state.keys,view)?,state::view(c,layout,&state.values,view)?))).collect::<Result<Vec<_>>>()?;
        let one=WindowOne {rows,heads,kv_heads,view,extent,scale:score_scale(a.head_dim),query:&products.0,query_scale,
            query_bound:a.query_key_clip as f32,moved:&products.1,moved_scale,moved_bound:a.relative.clip as f32,position:factor.as_ref(),
            keys:&planes[0].0,values:&planes[0].1,written:None,rel_table:table,sink:&sink,gate:&gate,steps:&both};
        let one_planes=if counts.iter().all(|&n|n<=spark_gpu_probe::flash_window::FW_ONE_VIEW_MAX) {c.window_planes(&one,&planes)?} else {None};
        if let Some(output)=one_planes {parts.push(output);} else {
            let attend=|s,values:bool,x:[&C::Value;3]|->Result<C::Value>{
                let views=states.iter().map(|state|state::view(c,layout,if values {&state.values} else {&state.keys},view)).collect::<Result<Vec<_>>>()?;
                if let Some(y)=c.block_planes(s,x,&views)? {return Ok(y);}
                // Backends without device plane tables consume the same per-chat slices.
                let part=|v:&C::Value,i:usize|{let width=c.words(v)/rows;c.range(v,i*width,width)};
                let ys=views.iter().enumerate().map(|(i,plane)|c.operation(s,[Some(&part(x[0],i)?),Some(plane),Some(&part(x[1],i)?),Some(&part(x[2],i)?)],&[])).collect::<Result<Vec<_>>>()?;
                c.join(&ys.iter().collect::<Vec<_>>())
            };
            let live=state::step_words(c,&counts)?;
            let (query,moved)=clipped()?;
            let mut s=shape(Op::RelBias);s.heads=number(heads)?;s.dim=number(relative)?;s.cols=number(extent)?;
            let bias=rows_operation(c,s,rows,[Some(&moved),None,None,None],0,table)?;
            let mut s=shape(Op::WindowScores);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(heads/kv_heads)?;s.rows=number(view)?;s.cols=number(extent)?;s.bound=score_scale(a.head_dim);
            let scores=attend(s,false,[&query,&bias,&both])?;
            let mut s=shape(Op::SinkSoftmax);s.heads=number(heads)?;s.rows=number(view)?;
            let weights=rows_operation(c,s,rows,[Some(&scores),Some(&sink),Some(&live),None],0,&[])?;
            let mut s=shape(Op::WindowValues);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(heads/kv_heads)?;s.rows=number(view)?;
            parts.push(attend(s,true,[&weights,&gate,&both])?);
        }
    } else {for (((span,state),scratch),fit) in spans.iter().zip(states.iter()).zip(&scratch).zip(&fits) {
        let wide=fit.wide();
        let (layout,planes)=match scratch {Some([keys,values])=>(&wide,[keys,values]),None=>(layout,[&state.keys,&state.values])};
        // Fast GQA-packed Split-KV path for decode/verify (`!prefill`): runs BEFORE any provisional sliding-window
        // ring copy (`state.written`), reading committed keys from `planes` and provisional rows directly from `delta`!
        if !prefill {
            let split_view = if layout.ring {
                layout.rows
            } else {
                layout.view_rows_in(span.first + span.count as u64 - 1, state::plane_rows(c, layout, planes[0])?)?
            };
            let (base_k, base_v) = (state::view(c, layout, planes[0], split_view)?, state::view(c, layout, planes[1], split_view)?);
            let mut split_done = 0usize;
            let mut split_parts = Vec::new();
            let mut split_ok = true;
            while split_done < span.count {
                let (n, at) = (spark_gpu_probe::flash_window::FW_SPLIT_ROWS_MAX.min(span.count - split_done), offset + split_done);
                // When provisional `state.written` holds > 16 rows, fall back rather than slicing delta past first_pos.
                if state.written.is_some() && span.count > spark_gpu_probe::flash_window::FW_SPLIT_ROWS_MAX {
                    split_ok = false; break;
                }
                let part = |x: &C::Value, words: usize| c.range(x, at * words, n * words);
                let first_p = span.first + split_done as u64;
                let both = key_steps_both(c, layout, first_p, n)?;
                let mut rpl = Vec::with_capacity(n);
                for r in 0..n {
                    let p = first_p + r as u64;
                    rpl.push((p, layout.live(p)?));
                }
                let position = match &factor { Some(f) => Some(part(f, 1)?), None => None };
                let delta = state.written.as_ref().map(|[dk, dv]| (dk, dv));
                let split_req = WindowSplit {
                    base: WindowOne {rows:n,heads,kv_heads,view:split_view,extent,scale:score_scale(a.head_dim),
                        query:&part(&products.0,heads*head)?,query_scale,query_bound:a.query_key_clip as f32,
                        moved:&part(&products.1,heads*relative)?,moved_scale,moved_bound:a.relative.clip as f32,
                        position:position.as_ref(),keys:&base_k,values:&base_v,written:None,rel_table:table,
                        sink:&part(&sink,heads)?,gate:&part(&gate,heads*head)?,steps:&both},
                    is_ring: layout.ring, first_pos: first_p, row_pos_live: &rpl, delta, planes: &[],
                };
                match c.window_split(&split_req)? {
                    Some(y) => {
                        need(c.words(&y) == n * heads * head, "the split-kv attention differs from the description")?;
                        split_parts.push(y);
                        split_done += n;
                    }
                    None => { split_ok = false; break; }
                }
            }
            if split_ok && split_done == span.count {
                parts.extend(split_parts);
                offset += span.count;
                continue;
            }
        }
        // A verify pass reads the live prefix and the proposed rows in chronological order.
        // Its original ring stays intact until accept, even when the proposed writes wrap.
        let group=heads/kv_heads;
        // Zero-copy provisional window verify: read dist <= row from `state.written` and dist > row from unmodified
        // `state.keys`/`state.values`, avoiding copying the 1,023-row sliding ring on every verify pass past wrap.
        if !prefill && let Some([wk,wv])=&state.written && span.count<=spark_gpu_probe::flash_window::FW_ONE_ROWS_MAX
            && layout.live(span.first+span.count as u64-1)?<=spark_gpu_probe::flash_window::FW_ONE_VIEW_MAX {
            let n=span.count;let part=|x:&C::Value,words:usize|c.range(x,offset*words,n*words);
            let both=key_steps_both(c,layout,span.first,n)?;
            let position=match &factor {Some(f)=>Some(part(f,1)?),None=>None};
            let one=WindowOne {rows:n,heads,kv_heads,view:layout.rows,extent,scale:score_scale(a.head_dim),query:&part(&products.0,heads*head)?,query_scale,
                query_bound:a.query_key_clip as f32,moved:&part(&products.1,heads*relative)?,moved_scale,moved_bound:a.relative.clip as f32,
                position:position.as_ref(),keys:planes[0],values:planes[1],written:Some((wk,wv)),rel_table:table,sink:&part(&sink,heads)?,gate:&part(&gate,heads*head)?,steps:&both};
            if let Some(y)=c.window_one(&one)? {
                need(c.words(&y)==n*heads*head,"the one-launch attention differs from the description")?;
                parts.push(y);offset+=span.count;continue;
            }
        }
        let (ring_delta, first, view) = if let Some([k, v]) = &state.written {
            let prefix = span.first.min((layout.window - 1) as u64) as usize;
            let start_slot = layout.slot(span.first - prefix as u64)?;
            let rd = (prefix > 0).then_some((k, v, prefix, start_slot));
            (rd, prefix as u64, prefix + span.count)
        } else {
            (None, span.first, layout.view_rows_in(span.first + span.count as u64 - 1, state::plane_rows(c, layout, planes[0])?)?)
        };
        let mut materialized: Option<(C::Value, C::Value)> = None;
        let mut views = |planes: [&C::Value; 2]| -> Result<(C::Value, C::Value)> {
            if materialized.is_none() {
                materialized = Some(if let Some([k, v]) = &state.written {
                    let prefix = span.first.min((layout.window - 1) as u64) as usize;
                    let slot = layout.slot(span.first - prefix as u64)?;
                    let n = prefix.min(layout.rows - slot);
                    let append = |plane: &C::Value, delta: &C::Value| {
                        let mut parts = Vec::new();
                        let w = layout.row_words;
                        if n > 0 { parts.push(c.range(plane, slot * w, n * w)?); }
                        if prefix > n { parts.push(c.range(plane, 0, (prefix - n) * w)?); }
                        let mut refs = parts.iter().collect::<Vec<_>>();
                        refs.push(delta);
                        c.join(&refs)
                    };
                    (append(planes[0], k)?, append(planes[1], v)?)
                } else {
                    (state::view(c, layout, planes[0], view)?, state::view(c, layout, planes[1], view)?)
                });
            }
            let (kv, vv) = materialized.as_ref().ok_or("loop decoder: attention views missing")?;
            Ok((c.range(kv, 0, c.words(kv))?, c.range(vv, 0, c.words(vv))?))
        };
        let view_layout=if state.written.is_some() {Layout {rows:view,..*layout}} else {*layout};
        // A check pass (and any step of rows outside a prompt): the attention of up to `FW_ONE_ROWS_MAX` rows in one
        // launch, the words of the arms below (`Compute::window_one`). A refused launch leaves the rest to the arms.
        let mut done=0;
        // A check pass past the first wrap has a chronological view of `window - 1 + count` rows, wider than the
        // one-launch form takes (`FW_ONE_VIEW_MAX`). Each row still sees at most `window` keys, so each row gets its
        // own slice of the view (the plane table of `Compute::window_planes`): the `FW_ONE_VIEW_MAX` rows that end at
        // its own key. Its live count is the same, its key at distance d is the same row of the view, and the sums run
        // by distance, so the words are those of the whole view. Without the slices every window layer of such a pass
        // ran the four separate arms.
        let one_max=spark_gpu_probe::flash_window::FW_ONE_VIEW_MAX;
        let sliced=state.written.is_some() && view>one_max && layout.window<=one_max;
        if !prefill && (ring_delta.is_none() || sliced || view<=one_max) {
            let (key_view,value_view)=views(planes)?;
            while done<span.count {
                let (n,at)=(spark_gpu_probe::flash_window::FW_ONE_ROWS_MAX.min(span.count-done),offset+done);
                if !sliced && view_layout.live(first+(done+n-1) as u64)?>one_max {break;}
                let part=|x:&C::Value,words:usize|c.range(x,at*words,n*words);
                let (both,slices)=if sliced {
                    // `first` is the row of the span's first key in the chronological view.
                    let own=usize::try_from(first).map_err(|_|"loop decoder: a check row exceeds this host".to_owned())?+done;
                    let rows=check_slices(own,n,view,layout.window,one_max).ok_or("loop decoder: a check row's slice differs from its window")?;
                    let (w,mut steps,mut slices)=(layout.row_words,Vec::with_capacity(2*n),Vec::with_capacity(n));
                    for (start,live,slot) in rows {
                        steps.extend([live,slot]);
                        slices.push((c.range(&key_view,start*w,one_max*w)?,c.range(&value_view,start*w,one_max*w)?));
                    }
                    (state::step_words(c,&steps)?,slices)
                } else {(key_steps_both(c,&view_layout,first+done as u64,n)?,Vec::new())};
                let position=match &factor {Some(f)=>Some(part(f,1)?),None=>None};
                let (keys,values,view)=match slices.first() {Some((k,v))=>(k,v,one_max),None=>(&key_view,&value_view,view)};
                let one=WindowOne {rows:n,heads,kv_heads,view,extent,scale:score_scale(a.head_dim),query:&part(&products.0,heads*head)?,query_scale,
                    query_bound:a.query_key_clip as f32,moved:&part(&products.1,heads*relative)?,moved_scale,moved_bound:a.relative.clip as f32,
                    position:position.as_ref(),keys,values,written:None,rel_table:table,sink:&part(&sink,heads)?,gate:&part(&gate,heads*head)?,steps:&both};
                let y=if sliced {c.window_planes(&one,&slices)?} else {c.window_one(&one)?};
                let Some(y)=y else {break};
                need(c.words(&y)==n*heads*head,"the one-launch attention differs from the description")?;
                parts.push(y);done+=n;
            }
        }
        if done==span.count {offset+=span.count;continue;}
        // A prompt: the tensor-core attention with the clips of the query and of the relative state inside the launch
        // (`Compute::window_rows` or `Weights::window_rows_half` emitting packed BF16 directly for `Weight::Output`),
        // the words of the clip launches and then the launch below.
        if prefill && state.written.is_none() {
            let (key_view,value_view)=views(planes)?;
            if spans.len() == 1 && w.prefers_half_output(layer) {
                let (mut fused_parts, mut done, mut ok) = (Vec::new(), 0, true);
                while done < span.count {
                    let (n, at) = (spark_gpu_probe::flash_window::FW_ROWS_MAX.min(span.count - done), offset + done);
                    let both = key_steps_both(c, layout, span.first + done as u64, n)?;
                    let part = |x: &C::Value, words: usize| c.range(x, at * words, n * words);
                    let position = match &factor {Some(f)=>Some(part(f,1)?),None=>None};
                    let one = WindowOne {rows:n,heads,kv_heads,view,extent,scale:score_scale(a.head_dim),query:&part(&products.0,heads*head)?,query_scale,
                        query_bound:a.query_key_clip as f32,moved:&part(&products.1,heads*relative)?,moved_scale,moved_bound:a.relative.clip as f32,
                        position:position.as_ref(),keys:&key_view,values:&value_view,written:None,rel_table:table,sink:&part(&sink,heads)?,gate:&part(&gate,heads*head)?,steps:&both};
                    match w.window_rows_half(c, &one, span.first + done as u64)? {
                        Some(out) => { need(c.words(&out)==n*heads*head/2,"the half prompt attention differs from the description")?; fused_parts.push(out); done += n; }
                        None => { ok = false; break; }
                    }
                }
                if ok {
                    parts.extend(fused_parts);
                    attended_half = true;
                    offset += span.count;
                    continue;
                }
            }
            let (mut fused_parts, mut done, mut ok) = (Vec::new(), 0, true);
            while done < span.count {
                let (n, at) = (spark_gpu_probe::flash_window::FW_ROWS_MAX.min(span.count - done), offset + done);
                let both = key_steps_both(c, layout, span.first + done as u64, n)?;
                let part = |x: &C::Value, words: usize| c.range(x, at * words, n * words);
                let position = match &factor {Some(f)=>Some(part(f,1)?),None=>None};
                let one = WindowOne {rows:n,heads,kv_heads,view,extent,scale:score_scale(a.head_dim),query:&part(&products.0,heads*head)?,query_scale,
                    query_bound:a.query_key_clip as f32,moved:&part(&products.1,heads*relative)?,moved_scale,moved_bound:a.relative.clip as f32,
                    position:position.as_ref(),keys:&key_view,values:&value_view,written:None,rel_table:table,sink:&part(&sink,heads)?,gate:&part(&gate,heads*head)?,steps:&both};
                match c.window_rows(&one, span.first + done as u64)? {
                    Some(out) => { need(c.words(&out)==n*heads*head,"the prompt attention differs from the description")?; fused_parts.push(out); done += n; }
                    None => { ok = false; break; }
                }
            }
            if ok {
                parts.extend(fused_parts);
                offset += span.count;
                continue;
            }
        }
        let (query,moved)=clipped()?;
        // Fused SM121a BF16 tensor-core attention (`FLASH_WINDOW_ATTN`): bypasses `RelBias + WindowScores +
        // SinkSoftmax + WindowValues` and avoids materializing `[n, heads, view]` score/weight buffers.
        // Split spans above `FW_ROWS_MAX` into fused launches so prompt cut length never changes the arithmetic path.
        if prefill && state.written.is_none() {
            let (key_view,value_view)=views(planes)?;
            let (mut fused_parts, mut done, mut ok) = (Vec::new(), 0, true);
            while done < span.count {
                let (n, at) = (spark_gpu_probe::flash_window::FW_ROWS_MAX.min(span.count - done), offset + done);
                let both = key_steps_both(c, layout, span.first + done as u64, n)?;
                let part = |x: &C::Value, words: usize| c.range(x, at * words, n * words);
                match c.flash_window_attn(
                    n, heads, kv_heads, head, relative, view, extent, score_scale(a.head_dim), span.first + done as u64,
                    &part(&query, heads * head)?, &key_view, &value_view, &part(&moved, heads * relative)?,
                    table, &part(&sink, heads)?, &part(&gate, heads * head)?, &both,
                )? {
                    Some(out) => { fused_parts.push(out); done += n; }
                    None => { ok = false; break; }
                }
            }
            if ok {
                parts.extend(fused_parts);
                offset += span.count;
                continue;
            }
        }
        // Bytes of one row over keys: the scores and the weights of each head over the view, and its relative term.
        let each=(key_bytes/4usize.saturating_mul(heads).saturating_mul(view.saturating_mul(2).saturating_add(extent)).max(1)).max(1);
        while done<span.count {
            let (n,at)=(each.min(span.count-done),offset+done);
            let part=|x:&C::Value,words:usize|c.range(x,at*words,n*words);
            let (both,live)=key_steps(c,&view_layout,first+done as u64,n)?;
            let mut s=shape(Op::RelBias);s.heads=number(heads)?;s.dim=number(relative)?;s.cols=number(extent)?;
            let bias=rows_operation(c,s,n,[Some(&part(&moved,heads*relative)?),None,None,None],0,table)?;
            let mut s=shape(Op::WindowScores);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(group)?;s.rows=number(view)?;s.cols=number(extent)?;s.bound=score_scale(a.head_dim);
            let q_part=part(&query,heads*head)?;
            let scores=if let Some((k,_,prefix,start_slot))=ring_delta
                && let Some(sc)=c.block_ring_delta(s,n,shared(1),[&q_part,&bias,&both],planes[0],k,layout.rows,start_slot,prefix)? {
                sc
            } else {
                let (key_view,_)=views(planes)?;
                rows_operation(c,s,n,[Some(&q_part),Some(&key_view),Some(&bias),Some(&both)],shared(1),&[])?
            };
            let mut s=shape(Op::SinkSoftmax);s.heads=number(heads)?;s.rows=number(view)?;
            let weights=rows_operation(c,s,n,[Some(&scores),Some(&part(&sink,heads)?),Some(&live),None],0,&[])?;
            let mut s=shape(Op::WindowValues);s.heads=number(heads)?;s.dim=number(head)?;s.taps=number(group)?;s.rows=number(view)?;
            let g_part=part(&gate,heads*head)?;
            let val=if let Some((_,v,prefix,start_slot))=ring_delta
                && let Some(out)=c.block_ring_delta(s,n,shared(1),[&weights,&g_part,&both],planes[1],v,layout.rows,start_slot,prefix)? {
                out
            } else {
                let (_,value_view)=views(planes)?;
                rows_operation(c,s,n,[Some(&weights),Some(&value_view),Some(&g_part),Some(&both)],shared(1),&[])?
            };
            parts.push(val);
            done+=n;
        }
        offset+=span.count;
    }}
    // The window rings after the step wait in the states as pending (`commit` puts them in).
    for (((span,state),scratch),fit) in spans.iter().zip(states.iter_mut()).zip(scratch).zip(&fits) {
        if let Some([keys,values])=scratch {let end=span.first+span.count as u64;state.planes=Some([state::many::narrow(c,fit,&keys,end)?,state::many::narrow(c,fit,&values,end)?]);}
    }
    let attended=if parts.len()==1 {parts.remove(0)} else {c.join(&parts.iter().collect::<Vec<_>>())?};
    let expected_attended=if attended_half {rows*heads*head/2} else {rows*heads*head};
    need(c.words(&attended)==expected_attended,"the attention output differs from the description")?;
    // Steps 6 and 7. The output product is the new input of the ring after attention; one launch gives the convolution
    // and the first sum.
    let output=if attended_half {
        let y=w.product_rows_half_in(c,Weight::Output(layer),&attended,rows)?
            .ok_or_else(||"loop decoder: half-input output product unavailable".to_owned())?;
        need(c.words(&y)==rows*width,"an attention product differs from the description")?;
        y
    } else {
        let y=w.product_rows(c,Weight::Output(layer),&attended,rows)?;
        need(c.words(&y)==rows*width,"an attention product differs from the description")?;
        y
    };
    let taps=w.words(Weight::ConvAfterAttention(layer))?;need(taps.len()==width*layout.taps,"the taps after attention differ from the description")?;
    let mut s=shape(Op::ShortConv);s.heads=1;s.dim=number(width)?;s.taps=number(layout.taps)?;s.bound=conv_scale(layout.taps as u64+1);
    if !spec.boundary {s.cols=CONV_ADDEND;}
    convolve(c,layout,Ring::AfterAttention,step,states,s,&output,None,(!spec.boundary).then_some(keep),taps)
}
/// The slice of each of `count` check rows in a chronological view of `view` rows whose own keys are the view rows
/// `own, own + 1, ...`: `(start, live, slot)`, the first view row of its `rows` rows, its live count
/// (`min(own + 1, window)`, as `Layout::live` of a ring) and its own key's row in the slice. The key at distance d
/// (`d < live`) is slice row `slot - d`, which is view row `own - d`, as in the whole view, and no slice row wraps.
/// None when a row's window does not fit in `rows` rows or a slice ends past the view.
pub(super) fn check_slices(own:usize,count:usize,view:usize,window:usize,rows:usize)->Option<Vec<(usize,usize,usize)>> {
    (own..own.checked_add(count)?).map(|own|{
        let (start,live)=((own+1).saturating_sub(rows),(own+1).min(window));
        (own<view && live<=own+1-start && start.checked_add(rows)?<=view).then_some((start,live,own-start))
    }).collect()
}
/// The largest count that divides both (0 and n give n).
fn common(a:usize,b:usize)->usize {if b==0 {a} else {common(b,a%b)}}
#[cfg(test)]
mod tests;
