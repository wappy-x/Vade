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
use crate::{Result,bounded_decoder::Compute};
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
    let source={let mut parts=states.iter().map(|state|&state.rings[which as usize]).collect::<Vec<_>>();parts.push(inputs);compute.join(&parts)?};
    let rings=permute(compute,&source,chats*taps+rows,&table.rows,channels)?;
    // Where every chat has one row, the ring of a row is the ring of its chat after the step.
    let after=if rows==chats {None} else {Some(permute(compute,&source,chats*taps+rows,&table.after,channels)?)};
    for (chat,state) in states.iter_mut().enumerate() {
        state.next[which as usize]=Some(compute.range(after.as_ref().unwrap_or(&rings),chat*taps*channels,taps*channels)?);
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
    let window={
        let ring=&states[0].rings[which as usize];
        // Position `first - (taps - 1) + j` lies in slot `(first + 1 + j) mod taps` of the ring.
        let mut parts=(0..taps-1).map(|j|compute.range(ring,((first+1+j)%taps)*channels,channels)).collect::<Result<Vec<_>>>()?;
        parts.push(compute.range(inputs,0,rows*channels)?);
        compute.join(&parts.iter().collect::<Vec<_>>())?
    };
    // The ring after the step: slot `s` holds the newest position with that slot, `back` rows before the last row,
    // which is row `rows + taps - 2` of the window.
    let last=(first+rows-1)%taps;
    let after=(0..taps).map(|slot|rows+taps-2-(last+taps-slot)%taps).collect::<Vec<_>>();
    states[0].next[which as usize]=Some(permute(compute,&window,rows+taps-1,&after,channels)?);
    Ok((window,step_words(compute,&[taps-1])?,spark_gpu_probe::block_rows::WINDOW|1<<3))
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
        }
        state.position+=span.count as u64;
    }
    Ok(())
}
/// The step did not end: what is pending goes. The rings, the planes and the positions are those from before the step.
pub fn discard<V>(states:&mut [&mut State<V>]) {
    for state in states {for visit in &mut state.visits {visit.next=std::array::from_fn(|_|None);visit.planes=None;}}
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
