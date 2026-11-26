//! The state in a step of many rows (`rows::Rows`): piece D1 of the rows work. A row is one position of one chat.
//! The rows of one chat in a step are consecutive positions, and the rows of a step stand in the order of its chats.
//!
//! Three things differ from a step of one row.
//! - **A convolution ring is not written.** Each row gets a ring of its own, made by one gather launch
//!   (`prompt_ops::Op::PermuteRows`) from the rings of the chats and the new inputs of all rows. Ring slot `s` of
//!   the row at position `q` holds the input of position `q - ((q - s) mod taps)`: the new input of that position
//!   if it is a row of this step, else slot `s` of the chat's ring. So an arm reads the ring words that it reads in
//!   a step of one row. `gather` makes the table, `row_rings` runs it.
//! - **The ring of a chat after the step waits.** It is the ring of the chat's last row. It is kept as pending in
//!   `VisitState::next`, and `commit` puts it into the state when the whole step is done. Until then the state holds
//!   the rings from before the step, so a step that fails can run again (`discard` drops what is pending).
//!   A pending ring is a value of its own, made by a second small gather: it is no view of the rings of all rows,
//!   which would stay alive with it.
//! - **A plane takes the rows of a chat in one write**, or in two where a ring goes round (`write_rows`).
use super::{Layout,Ring,State,VisitState,need,step_words};
use super::super::rows::{ChatRows,Rows};
use crate::{Result,bounded_decoder::{Compute,PrefillKv}};
use spark_gpu_probe::prompt_ops;

/// The table of one ring gather. The source rows are the ring of each chat in step order (`taps` rows a chat), then
/// the new input of each row of the step.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Gather {
    /// `[rows][taps]`: the source row of each ring slot of each row.
    pub rows:Vec<usize>,
    /// `[rows]`: the ring slot of the row's own position. This is the step word of arm `ShortConv`.
    pub slots:Vec<usize>,
    /// `[chats][taps]`: the source rows of the ring of each chat after the step.
    pub after:Vec<usize>,
}
/// The gather table of a step for rings of `taps` rows.
pub fn gather(spans:&[ChatRows],taps:usize)->Result<Gather> {
    need(taps>0 && !spans.is_empty() && spans.iter().all(|span|span.count>0),"a step needs a chat, and each chat a row")?;
    let (chats,rows)=(spans.len(),spans.iter().map(|span|span.count).sum::<usize>());
    let mut out=Gather {rows:Vec::with_capacity(rows*taps),slots:Vec::with_capacity(rows),after:Vec::with_capacity(chats*taps)};
    let mut offset=0;
    for (chat,span) in spans.iter().enumerate() {
        let first=(span.first%taps as u64) as usize;
        for row in 0..span.count {
            let own=(first+row%taps)%taps;out.slots.push(own);
            // Slot `slot` holds the newest position up to the row's own that has this slot: `back` rows before it.
            for slot in 0..taps {let back=(own+taps-slot)%taps;out.rows.push(if back<=row {chats*taps+offset+row-back} else {chat*taps+slot});}
        }
        let last=offset+span.count-1;out.after.extend_from_slice(&out.rows[last*taps..(last+1)*taps]);
        offset+=span.count;
    }
    Ok(out)
}
/// One gather launch: output row `i` is row `table[i]` of `source`. A row has `width` words.
fn permute<C:Compute>(compute:&C,source:&C::Value,source_rows:usize,table:&[usize],width:usize)->Result<C::Value> {
    let number=|n:usize|u32::try_from(n).map_err(|_|"loop decoder state: a gather exceeds the launch format".to_owned());
    let mut shape=prompt_ops::Shape::new(prompt_ops::Op::PermuteRows,1).ok_or("loop decoder state: a gather has no shape")?;
    shape.rows=number(table.len())?;shape.cols=number(width)?;shape.first=number(source_rows)?;
    compute.prompt(shape,[Some(source),Some(&step_words(compute,table)?),None,None,None],&[])
}
/// The ring that each row of a step reads in convolution `which`, and the step word of each row.
/// `inputs` is `[rows][channels]`: the new input of each row, in step order. `states` holds the state of this visit
/// of each chat, in step order. The result is `[rows][taps][channels]` and `[rows]`.
/// The ring of each chat after the step becomes pending. No ring of the state changes here.
pub fn row_rings<C:Compute>(compute:&C,layout:&Layout,which:Ring,step:&Rows,states:&mut [&mut VisitState<C::Value>],inputs:&C::Value)->Result<(C::Value,C::Value)> {
    let ((channels,taps),rows,chats)=(layout.ring_size(which),step.total(),step.chats().len());
    need(states.len()==chats && compute.words(inputs)==rows*channels && states.iter().all(|state|compute.words(&state.rings[which as usize])==taps*channels),
        "the rings or the inputs of a step differ from the layout")?;
    let table=gather(step.chats(),taps)?;
    // Where every chat has one row, the ring of a row is the ring of its chat after the step.
    let (rings,after)=if rows==chats {
        // Every chat has one row: the ring of its row is the chat's ring with the row's own slot taken by its input
        // (the gather table names that input for the own slot and the chat's ring slot for every other). The same
        // words in one join, with no gather launch.
        let mut parts=Vec::with_capacity(3*chats);
        for (chat,(state,&own)) in states.iter().zip(&table.slots).enumerate() {
            let ring=&state.rings[which as usize];
            if own>0 {parts.push(compute.range(ring,0,own*channels)?);}
            parts.push(compute.range(inputs,chat*channels,channels)?);
            if own+1<taps {parts.push(compute.range(ring,(own+1)*channels,(taps-own-1)*channels)?);}
        }
        (compute.join(&parts.iter().collect::<Vec<_>>())?,None)
    } else {
        let source={let mut parts=states.iter().map(|state|&state.rings[which as usize]).collect::<Vec<_>>();parts.push(inputs);compute.join(&parts)?};
        let rings=permute(compute,&source,chats*taps+rows,&table.rows,channels)?;
        let after=if step.keeps() {None} else {Some(permute(compute,&source,chats*taps+rows,&table.after,channels)?)};
        (rings,after)
    };
    if step.keeps() {
        // A step that keeps each row (`keeping` / `keeping_many`): every chat takes its kept rings directly from `rings`.
        for (chat,state) in states.iter_mut().enumerate() {
            let (off,count)=(step.offset(chat),step.chats()[chat].count);
            state.next[which as usize]=Some(compute.range(&rings,(off+count-1)*taps*channels,taps*channels)?);
            state.every[which as usize]=Some(compute.range(&rings,off*taps*channels,count*taps*channels)?);
        }
    } else {
        for (chat,state) in states.iter_mut().enumerate() {
            state.next[which as usize]=Some(compute.range(after.as_ref().unwrap_or(&rings),chat*taps*channels,taps*channels)?);
        }
    }
    Ok((rings,step_words(compute,&table.slots)?))
}
/// The rings of a convolution for a launch of arm `ShortConv`: the input 0, the step word input 3, and the mask
/// bits of the rows launch. A step of more than one row of ONE chat takes the window form of the rows entry
/// (`spark_gpu_probe::block_rows::WINDOW`): the `taps - 1` positions before the step from the chat's ring in time
/// order, then the new inputs, in one join; row `r` reads rows `r ..= r + taps - 1` of it, and one shared step word
/// `taps - 1`. So no ring is gathered for each row. The ring of the chat after the step becomes pending, made by one
/// small gather as in `row_rings`. Any other step takes `row_rings` and the mask 0. No ring of the state changes here.
pub fn conv_rings<C:Compute>(compute:&C,layout:&Layout,which:Ring,step:&Rows,states:&mut [&mut VisitState<C::Value>],inputs:&C::Value)->Result<(C::Value,C::Value,u32)> {
    let ((channels,taps),rows)=(layout.ring_size(which),step.total());
    let [span]=step.chats() else {let (rings,steps)=row_rings(compute,layout,which,step,states,inputs)?;return Ok((rings,steps,0));};
    if rows<2 || taps<2 {let (rings,steps)=row_rings(compute,layout,which,step,states,inputs)?;return Ok((rings,steps,0));}
    need(states.len()==1 && compute.words(inputs)==rows*channels && compute.words(&states[0].rings[which as usize])==taps*channels,
        "the rings or the inputs of a step differ from the layout")?;
    let first=(span.first%taps as u64) as usize;
    let window=window(compute,&states[0].rings[which as usize],first,taps,channels,inputs,rows)?;
    // The ring after the step: slot `s` holds the newest position with that slot, `back` rows before the last row,
    // which is row `rows + taps - 2` of the window.
    let last=(first+rows-1)%taps;
    let after=(0..taps).map(|slot|rows+taps-2-(last+taps-slot)%taps).collect::<Vec<_>>();
    if step.keeps() {
        // A step that keeps each row: the ring after each row `r` in one gather, the same rule with row `r + taps - 1`
        // of the window as the newest. The ring after the last row is its last part.
        let every=(0..rows).flat_map(|row|{let own=(first+row)%taps;(0..taps).map(move |slot|row+taps-1-(own+taps-slot)%taps)}).collect::<Vec<_>>();
        let all=permute(compute,&window,rows+taps-1,&every,channels)?;
        states[0].next[which as usize]=Some(compute.range(&all,(rows-1)*taps*channels,taps*channels)?);
        states[0].every[which as usize]=Some(all);
    } else {states[0].next[which as usize]=Some(permute(compute,&window,rows+taps-1,&after,channels)?);}
    Ok((window,step_words(compute,&[taps-1])?,spark_gpu_probe::block_rows::WINDOW|1<<3))
}

/// The window of a step of `rows` rows of one chat: the `taps - 1` positions before the step from `ring` (position
/// `first - (taps - 1) + j` lies in slot `(first + 1 + j) mod taps`), then the inputs of the rows.
fn window<C:Compute>(compute:&C,ring:&C::Value,first:usize,taps:usize,channels:usize,inputs:&C::Value,rows:usize)->Result<C::Value> {
    let mut parts=(0..taps-1).map(|j|compute.range(ring,((first+1+j)%taps)*channels,channels)).collect::<Result<Vec<_>>>()?;
    parts.push(compute.range(inputs,0,rows*channels)?);
    compute.join(&parts.iter().collect::<Vec<_>>())
}
/// A prefill tile long enough to replace the ring can read the old ring and new
/// rows in place. Its pending ring then comes entirely from the new rows.
#[allow(clippy::too_many_arguments)]
pub fn convolve<C:Compute>(
    compute:&C, layout:&Layout, which:Ring, step:&Rows, states:&mut [&mut VisitState<C::Value>],
    shape:spark_gpu_probe::block_ops::Shape, inputs:&C::Value, token:Option<&C::Value>, addend:Option<&C::Value>, fixed:&[f32],
)->Result<C::Value> {
    let ((channels,taps),rows)=(layout.ring_size(which),step.total());
    if let ([span], [state])=(step.chats(),&mut *states) && rows>=taps && rows>1 && taps>1 {
        let first=(span.first%taps as u64) as usize;
        let ring=&state.rings[which as usize];
        if let Some(output)=compute.short_conv_prefill(shape,rows,first,[Some(inputs),token,addend,Some(ring)],fixed)? {
            state.next[which as usize]=Some(prefill_ring_after(compute,inputs,rows,channels,taps,first)?);
            return Ok(output);
        }
    }
    // A step that keeps each row (a single-chat or multi-chat check pass): one `FastStreamOps` launch reads `ring` +
    // `inputs` directly (zero `PermuteRows` / `block_rows`) and writes both the convolution output and every kept ring.
    if step.keeps() && rows>1 && taps>1 && compute.words(inputs)==rows*channels {
        if let ([span], [state])=(step.chats(),&mut *states) && compute.words(&state.rings[which as usize])==taps*channels {
            let first=(span.first%taps as u64) as usize;
            let ring=&state.rings[which as usize];
            if let Some((output,every))=compute.short_conv_keep(shape,rows,first,[Some(inputs),token,addend,Some(ring)],fixed)? {
                need(compute.words(&output)==rows*channels && compute.words(&every)==rows*taps*channels,"the kept convolution differs from the rows of the step")?;
                state.next[which as usize]=Some(compute.range(&every,(rows-1)*taps*channels,taps*channels)?);
                state.every[which as usize]=Some(every);
                return Ok(output);
            }
        } else if step.chats().len()>1 && step.chats().len()==states.len() && rows<=16 && taps<=4
            && states.iter().all(|state|compute.words(&state.rings[which as usize])==taps*channels)
        {
            let mut start_mask=0u16;
            let mut slots=0u32;
            for (chat,span) in step.chats().iter().enumerate() {
                start_mask|=1u16<<step.offset(chat);
                slots|=((span.first%taps as u64) as u32)<<(2*chat);
            }
            if let Some(packed)=spark_gpu_probe::fast_ops::short_conv_multi_encode(rows,taps,start_mask,slots) {
                let rings_joined=compute.join(&states.iter().map(|s|&s.rings[which as usize]).collect::<Vec<_>>())?;
                if let Some((output,every))=compute.short_conv_keep(shape,rows,packed,[Some(inputs),token,addend,Some(&rings_joined)],fixed)? {
                    need(compute.words(&output)==rows*channels && compute.words(&every)==rows*taps*channels,"the kept convolution differs from the rows of the step")?;
                    for (chat,state) in states.iter_mut().enumerate() {
                        let (off,count)=(step.offset(chat),step.chats()[chat].count);
                        state.next[which as usize]=Some(compute.range(&every,(off+count-1)*taps*channels,taps*channels)?);
                        state.every[which as usize]=Some(compute.range(&every,off*taps*channels,count*taps*channels)?);
                    }
                    return Ok(output);
                }
            }
        }
    }
    let (rings,steps,window)=conv_rings(compute,layout,which,step,states,inputs)?;
    super::super::rows::rows_operation(compute,shape,rows,[Some(&rings),token,addend,Some(&steps)],window,fixed)
}
fn prefill_ring_after<C: Compute>(
    compute: &C, inputs: &C::Value, rows: usize, channels: usize, taps: usize, first: usize,
) -> Result<C::Value> {
    let last = (first + rows - 1) % taps;
    let after = (0..taps).map(|slot| rows - 1 - (last + taps - slot) % taps).collect::<Vec<_>>();
    permute(compute, inputs, rows, &after, channels)
}

/// Finish one prefill tile in its writable KV plane. The pending ring is the same gather as `convolve`.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn convolve_kv<C: Compute>(
    compute: &C, which: Ring, span: ChatRows, shape: spark_gpu_probe::block_ops::Shape,
    input: &C::Value, made: Option<&C::Value>, fixed: &[f32], state: &mut VisitState<C::Value>,
    wide: Option<&mut C::Value>, write: spark_gpu_probe::fast_ops::KvWrite,
) -> Result<bool> {
    let (rows, taps) = (span.count, shape.taps as usize);
    if rows < taps || rows < 2 || taps < 2 { return Ok(false); }
    let ring_first = (span.first % taps as u64) as usize;
    let conv = PrefillKv {shape, rows, ring_first, ring: &state.rings[which as usize], input, made, fixed, write};
    let plane = wide.unwrap_or(if which == Ring::Key { &mut state.keys } else { &mut state.values });
    if !compute.prefill_kv(&conv, plane)? { return Ok(false); }
    let channels = shape.heads as usize * shape.dim as usize;
    state.next[which as usize] = Some(prefill_ring_after(compute,input,rows,channels,taps,ring_first)?);
    Ok(true)
}

/// The most rows of one chat that a step may write into a plane of `layout` from position `first`: a later row
/// must not write over a key that an earlier row of the step still reads, and a plane must hold every row.
pub fn room(layout:&Layout,first:u64)->usize {
    let left=(layout.rows as u64).saturating_sub(first).min(usize::MAX as u64) as usize;
    if layout.ring {left.max(layout.rows-layout.window+1)} else {left}
}
/// Put the finished keys or values of the rows of one chat into its plane. `packed` is `[count][row words]`: the
/// rows as `pack` gives them. One write, or two where a ring goes round. No other row changes.
pub fn write_rows<C:Compute>(compute:&C,layout:&Layout,plane:&mut C::Value,span:ChatRows,packed:&C::Value)->Result<()> {
    let words=layout.row_words;
    need(span.count>0 && compute.words(packed)==span.count*words && compute.words(plane)==layout.rows*words,"the rows of a key or value write differ from the layout")?;
    need(span.count<=room(layout,span.first),"a step has more rows of one chat than its state takes")?;
    let (slot,head)=(layout.slot(span.first)?,span.count.min(layout.rows-layout.slot(span.first)?));
    if head==span.count {return compute.state_fill(plane,slot*words,packed);}
    compute.state_fill(plane,slot*words,&compute.range(packed,0,head*words)?)?;
    compute.state_fill(plane,0,&compute.range(packed,head*words,(span.count-head)*words)?)
}
/// The step words of the attention arms for `count` rows of one chat from position `first`: `[count][2]`, the live
/// count and the slot of each row, and `[count]`, the live count alone.
pub fn key_steps<C:Compute>(compute:&C,layout:&Layout,first:u64,count:usize)->Result<(C::Value,C::Value)> {
    let (mut both,mut live)=(Vec::with_capacity(2*count),Vec::with_capacity(count));
    for position in (first..).take(count) {let n=layout.live(position)?;both.extend([n,layout.slot(position)?]);live.push(n);}
    Ok((step_words(compute,&both)?,step_words(compute,&live)?))
}
/// Only the `[count][2]` `(live, slot)` step words when the caller does not need the separate `live` vector.
pub fn key_steps_both<C:Compute>(compute:&C,layout:&Layout,first:u64,count:usize)->Result<C::Value> {
    let mut both=Vec::with_capacity(2*count);
    for position in (first..).take(count) {both.extend([layout.live(position)?,layout.slot(position)?]);}
    step_words(compute,&both)
}
/// The whole step is done: each pending ring becomes the ring of its chat, and each chat moves on by its rows.
/// `Driver::step_rows_seen` calls it one time, after the last visit and the end. Nothing else moves a position.
/// Every ring of every visit must have its pending value, else nothing changes and the call is refused.
pub fn commit<V>(states:&mut [&mut State<V>],step:&Rows)->Result<()> {
    let spans=step.chats();
    need(states.len()==spans.len() && states.iter().zip(spans).all(|(state,span)|state.position==span.first),"the states of a step differ from its rows")?;
    need(states.iter().all(|state|state.visits.iter().all(|visit|visit.next.iter().all(Option::is_some))),"a visit of the step left a ring without its rows")?;
    for (state,span) in states.iter_mut().zip(spans) {
        for visit in &mut state.visits {
            for (ring,next) in visit.rings.iter_mut().zip(&mut visit.next) {if let Some(next)=next.take() {*ring=next;}}
            if let Some([keys,values])=visit.planes.take() {visit.keys=keys;visit.values=values;}
            visit.every=std::array::from_fn(|_|None);visit.written=None;
        }
        state.position+=span.count as u64;
    }
    Ok(())
}
/// The step did not end: what is pending goes. The rings, the planes and the positions are those from before the step.
pub fn discard<V>(states:&mut [&mut State<V>]) {
    for state in states {for visit in &mut state.visits {visit.next=std::array::from_fn(|_|None);visit.planes=None;visit.every=std::array::from_fn(|_|None);visit.written=None;}}
}
/// The end of a step that kept each row (`rows::Rows::keeping`) after its first `count` rows of `rows`: the chat
/// takes the ring after row `count - 1` of every convolution, the window planes as they are after that row, and
/// moves on by `count`. The rows after it are as if they never ran: the rows of their positions in a straight plane
/// are written again by the next step before any row reads them (the rule of `Driver::step_rows_seen`).
/// A window plane: with every row taken, the pending plane of the step; else the plane from before the step with
/// the first `count` key and value rows written in at their slots, which puts over only rows that are out of the
/// window from position `first + count` on.
pub fn accept<C:Compute>(compute:&C,layouts:&[Layout],state:&mut State<C::Value>,rows:usize,count:usize)->Result<()> {
    need(count>0 && count<=rows && state.visits.len()==layouts.len(),"a kept step ends after one of its rows")?;
    need(state.visits.iter().all(|visit|visit.next.iter().all(Option::is_some) && visit.every.iter().all(Option::is_some)),"the state holds no kept step")?;
    let first=state.position;
    for (visit,layout) in state.visits.iter_mut().zip(layouts) {
        for (index,ring) in Ring::ALL.iter().enumerate() {
            let (channels,taps)=layout.ring_size(*ring);
            let (next,every)=(visit.next[index].take(),visit.every[index].take());
            visit.rings[index]=if count==rows {next.ok_or("loop decoder state: a ring has no pending value")?} else {
                let every=every.ok_or("loop decoder state: a ring kept no row")?;
                need(compute.words(&every)==rows*taps*channels,"the kept rings differ from the rows of the step")?;
                compute.range(&every,(count-1)*taps*channels,taps*channels)?
            };
        }
        let (planes,written)=(visit.planes.take(),visit.written.take());
        if count==rows && let Some([keys,values])=planes {visit.keys=keys;visit.values=values;continue;}
        let Some([keys,values])=written else {continue};
        let words=layout.row_words;
        need(layout.ring && compute.words(&keys)==rows*words && compute.words(&values)==rows*words,"the kept key and value rows differ from the layout")?;
        for (plane,packed) in [(&mut visit.keys,&keys),(&mut visit.values,&values)] {
            let mut done=0;
            while done<count {let slot=layout.slot(first+done as u64)?;let n=(count-done).min(layout.rows-slot);compute.state_fill(plane,slot*words,&compute.range(packed,done*words,n*words)?)?;done+=n;}
        }
    }
    state.position+=count as u64;
    Ok(())
}
/// The runs of rows that move positions `from..to` from a ring of `source` rows to a ring of `target` rows:
/// (source row, target row, rows), each run unbroken in both rings.
pub fn moves(from:u64,to:u64,source:usize,target:usize)->Vec<(usize,usize,usize)> {
    let (mut out,mut position)=(Vec::new(),from);
    while position<to {
        let (s,t)=((position%source as u64) as usize,(position%target as u64) as usize);
        let n=((to-position).min(usize::MAX as u64) as usize).min(source-s).min(target-t);
        out.push((s,t,n));position+=n as u64;
    }
    out
}
/// Copy the rows of `moves` from `source` into `target`, `words` words a row.
fn carry<C:Compute>(compute:&C,source:&C::Value,target:&mut C::Value,runs:&[(usize,usize,usize)],words:usize)->Result<()> {
    for &(s,t,n) in runs {compute.state_fill(target,t*words,&compute.range(source,s*words,n*words)?)?;}
    Ok(())
}
/// The wide ring (`Layout::wide`) of a plane of the state for a chat whose step starts at position `first`: the keys
/// of positions `first - window + 1` to `first - 1` at their wide slots, every other row zero. The state is read only.
pub fn widen<C:Compute>(compute:&C,layout:&Layout,plane:&C::Value,first:u64)->Result<C::Value> {
    let (wide,words)=(layout.wide(),layout.row_words);
    need(layout.ring && compute.words(plane)==layout.rows*words,"a plane to widen differs from the layout")?;
    let mut out=compute.zeros(wide.rows*words)?;
    carry(compute,plane,&mut out,&moves(first.saturating_sub(layout.window as u64-1),first,layout.rows,wide.rows),words)?;Ok(out)
}
/// The window ring of a wide plane after a step whose last position is `end - 1`: the keys of positions `end - window`
/// to `end - 1` at their slots, every other row zero. A new value: the state does not change here.
pub fn narrow<C:Compute>(compute:&C,layout:&Layout,wide_plane:&C::Value,end:u64)->Result<C::Value> {
    let (wide,words)=(layout.wide(),layout.row_words);
    need(layout.ring && compute.words(wide_plane)==wide.rows*words,"a wide plane differs from the layout")?;
    let mut out=compute.zeros(layout.rows*words)?;
    carry(compute,wide_plane,&mut out,&moves(end.saturating_sub(layout.window as u64),end,wide.rows,layout.rows),words)?;Ok(out)
}
#[cfg(test)]
mod tests;
