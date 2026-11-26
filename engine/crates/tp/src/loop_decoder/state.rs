//! The state of a chat for each layer visit: key and value rings or planes, and the convolution rings.
//! The slot of position `p` is `p mod rows` in a ring and `p` in a straight plane. A new chat starts from zeros.
//!
//! One visit keeps six values. A visit is one run of one layer for one token, so a layer that runs three times has
//! three visits and three states (reference lines 1354-1357 and 1405 of `transformers` v5.24.0,
//! `models/inkling2_loop/modeling_inkling2_loop.py`; this file follows them and holds its own code).
//! - The key plane and the value plane. One row a token: key heads times head size BF16 values, packed two a
//!   word as `Op::PackBf16` writes them. A row holds the FINISHED key or value: after the convolution, and after
//!   the clip (key) or the norm (value). A window layer has a ring of `window` rows, and of a few spare rows more
//!   in a driver that takes whole check passes (`spare_rows`). A layer that sees every token has a straight plane
//!   of `capacity` rows.
//! - Four convolution rings of `taps` rows each: for the key, the value, the stream after attention and the stream
//!   after the experts. A row holds one INPUT of its convolution (the vector before the convolution), as 32-bit words.
//!
//! Rules:
//! - A ring never starts anywhere else: the slot is a function of the position alone. So a state that was built in
//!   one run and a state that was copied out and back in hold each row in the same place.
//! - A token writes its own rows BEFORE its attention reads the planes. The attention arms then read every key
//!   from the state, the token's own key too.
//! - A token at position `p` writes only the slots of position `p`. No token at a position up to `p` can see
//!   what those slots held before (the row of position `p - rows`, the input of position `p - taps`). So a step that
//!   fails leaves a state that is still right for its position, and the same step can run again.
//! - A chat starts with every word zero: tokens before the start count as zero in a convolution, and no key is live.
//!
//! A count or a slot goes to an arm as one 32-bit word that holds the number (as the count word of the older
//! attention arms). `step_words` is the one place that makes such words.
//!
//! A step of many rows (`many`): a chat can take several positions in one step. The ring of a window layer in the
//! state keeps `window` rows, and with a step row limit of a whole check pass or more it keeps the spare rows of one
//! check pass too (`spare_rows`); a token step still reads only its window. A chat with more rows in a step than the
//! state ring takes (`many::room`) works in a wider ring of its own, `window + step row limit - 1` rows
//! (`Layout::wide`): the rows it needs from the state go in first (`many::widen`), a row of the step still sees
//! `window` keys at most, and no row of the step writes over a key that an earlier row of the step still reads. The
//! window ring after the step (`many::narrow`) waits as pending until the whole step is done (`commit`), so the
//! state keeps every key a step that fails needs to run again. The order of a sum over keys is by distance, so the
//! ring size changes no word. A convolution ring is not written in such a step: each row gets its ring by one
//! gather, and the ring after the step is kept as pending until the whole step is done (`commit`).
use crate::{Result,bounded_decoder::Compute};
use engine_model::plan::description::LoopTextDescription;
use super::rows::Limits;
use spark_gpu_probe::causal_ops::{BUCKET_ROWS,Op,Shape};

pub mod many;
pub use many::{accept,commit,conv_rings,discard,row_rings};

/// A 32-bit float holds every whole number up to this one. A larger count or slot cannot be a step word.
pub const STEP_WORD_LIMIT:usize=1<<24;
/// The most rows of one chat in a check pass of drafted tokens that go straight into a window ring: the newest token
/// and the proposed ones, the rows of one attention launch.
pub const CHECK_ROWS:usize=spark_gpu_probe::flash_window::FW_ONE_ROWS_MAX;
/// The spare rows of a window ring in the state, above its window, for a step row limit of `step_rows`: enough for
/// one whole check pass (`CHECK_ROWS - 1`) where the limit takes one, else none.
///
/// Why spare rows: the row of position `p` lies in slot `p mod rows`, where the row of position `p - rows` was. With
/// `rows = window + spare`, a step of up to `spare + 1` rows of a chat from position `first` writes only over
/// positions up to `first - window`. No row from position `first` on reads those (it sees `window` keys at most).
/// So the rows go into the ring before the chat keeps them: the rows that the chat does not keep are written again
/// by the next step before any row reads them, as in a straight plane, and many chats share one attention launch.
/// The order of a sum over keys is by position and by distance, never by slot, so the ring size changes no word.
///
/// Why none under `CHECK_ROWS`: a driver with a small limit keeps the ring of the token path (its window), and its
/// check passes keep their rows apart (`VisitState::written`).
pub const fn spare_rows(step_rows:usize)->usize {if step_rows>=CHECK_ROWS {CHECK_ROWS-1} else {0}}
fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(format!("loop decoder state: {why}"))}}
pub(super) fn size(n:u64)->Result<usize> {usize::try_from(n).map_err(|_|"loop decoder state: a size exceeds this host".to_owned())}

/// The four convolutions of one visit, in the order of the reference's state index (reference lines 562-567 and 868-873).
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Ring {Key,Value,AfterAttention,AfterExperts}
impl Ring {pub const ALL:[Self;4]=[Self::Key,Self::Value,Self::AfterAttention,Self::AfterExperts];}

/// The sizes of the state of one visit. Every number comes from the description of the visit's layer.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Layout {
    /// Rows of the key plane and of the value plane in the state. A window layer: its window and the spare rows of a
    /// check pass (`spare_rows`; `wide` gives the ring of a chat with more rows in a step). Else the admitted context.
    pub rows:usize,
    /// The most keys that one token sees: the window of a window layer, else `rows`.
    pub window:usize,
    /// The most rows of one step that this state takes (the step row limit).
    pub step_rows:usize,
    /// The most bytes of the values over the keys (relative terms, scores, weights) in one launch group.
    pub key_bytes:usize,
    /// True: the planes are rings. False: straight planes.
    pub ring:bool,
    /// Words of one packed row: key heads times head size, two BF16 values a word.
    pub row_words:usize,
    /// Channels and taps of the key and of the value convolution.
    pub kv_channels:usize,pub kv_taps:usize,
    /// Channels and taps of the convolution after attention and of the convolution after the experts.
    pub channels:usize,pub taps:usize,
}
impl Layout {
    /// The layout of a visit of stored layer `layer` for steps of one row. `capacity` is the longest chat that
    /// the state can hold.
    pub fn new(d:&LoopTextDescription,layer:usize,capacity:u64)->Result<Self> {Self::for_steps(d,layer,capacity,Limits::ONE)}
    /// The layout for steps of up to `limits.step_rows` rows.
    pub fn for_steps(d:&LoopTextDescription,layer:usize,capacity:u64,limits:Limits)->Result<Self> {
        let (step_rows,key_bytes)=(limits.step_rows,limits.key_bytes);
        let a=d.layers.get(layer).ok_or("loop decoder state: a visit names a layer outside the description")?.attention;
        let (kv_channels,capacity)=(size(a.kv_heads.checked_mul(a.head_dim).ok_or("loop decoder state: key width overflow")?)?,size(capacity)?);
        need(capacity>0 && capacity as u64<=d.max_positions,"the capacity is zero or above the model's longest chat")?;
        need(kv_channels>0 && kv_channels.is_multiple_of(2),"a packed row needs an even count of values")?;
        need(step_rows>0 && step_rows<=STEP_WORD_LIMIT && key_bytes>0,"a limit of a step is zero or too large")?;
        // The spare rows are at most `step_rows - 1`, so the checked sum bounds the ring of the state too.
        let (ring,rows,window)=match a.window {Some(window)=>{let window=size(window)?;window.checked_add(step_rows-1).ok_or("loop decoder state: a ring size overflows")?;(true,window+spare_rows(step_rows),window)},None=>(false,capacity,capacity)};
        let layout=Self {rows,window,step_rows,key_bytes,ring,row_words:kv_channels/2,kv_channels,kv_taps:size(a.key_value_conv.kernel)?,channels:size(d.hidden)?,taps:size(d.conv.kernel)?};
        let wide=layout.wide().rows;
        need(rows>0 && layout.kv_taps>0 && layout.taps>0 && layout.channels>0,"a state size is zero")?;
        need(wide<=STEP_WORD_LIMIT,"a plane has more rows than a step word can count")?;
        need(wide.checked_mul(layout.row_words).is_some() && layout.kv_taps.checked_mul(kv_channels).is_some() && layout.taps.checked_mul(layout.channels).is_some(),"a state size overflows")?;
        Ok(layout)
    }
    /// The layout of the planes that a chat with more than one row in a step works in: a window layer's ring of
    /// `window + step_rows - 1` rows, so no row of the step writes over a key that an earlier row still reads. A
    /// straight plane, or a step row limit of 1: the layout itself.
    pub fn wide(&self)->Self {if self.ring {Self {rows:self.window+self.step_rows-1,..*self}} else {*self}}
    /// The row of the token at `position` in the key plane and in the value plane.
    pub fn slot(&self,position:u64)->Result<usize> {
        if self.ring {return Ok((position%self.rows as u64) as usize);}
        need(position<self.rows as u64,"the chat is longer than the state's capacity")?;Ok(position as usize)
    }
    /// The keys that the token at `position` sees, its own key included.
    pub fn live(&self,position:u64)->Result<usize> {
        if self.ring {return Ok(position.saturating_add(1).min(self.window as u64) as usize);}
        need(position<self.rows as u64,"the chat is longer than the state's capacity")?;Ok(position as usize+1)
    }
    /// The rows of a plane that the attention arms get for the token at `position`. A ring: all of them, so the grid of
    /// its launches never changes. A straight plane: its first rows up to the next full bucket of the engine's rule for
    /// launches that grow with the context, so the grid changes once a bucket and not at every token.
    pub fn view_rows(&self,position:u64)->Result<usize> {
        if self.ring {return Ok(self.rows);}
        Ok(self.live(position)?.checked_next_multiple_of(BUCKET_ROWS as usize).map_or(self.rows,|rows|rows.min(self.rows)))
    }
    /// `view_rows` bounded by the plane's currently mapped rows `plane_rows`. On paged GPU planes (`1,024` rows/page,
    /// multiple of `BUCKET_ROWS = 256`), `view_rows(position) <= plane_rows` always holds; on tiny host fixtures with
    /// `< 256` rows/page, this keeps the view inside the mapped prefix.
    pub fn view_rows_in(&self,position:u64,plane_rows:usize)->Result<usize> {
        if self.ring {
            need(plane_rows==self.rows,"a ring plane differs from its window")?;
            return Ok(self.rows);
        }
        let live=self.live(position)?;
        need(live<=plane_rows && plane_rows<=self.rows,"a straight plane has fewer mapped rows than live tokens")?;
        Ok(self.view_rows(position)?.min(plane_rows))
    }
    /// Rows of a straight KV plane mapped for `tokens` tokens at `page_bytes` allocation granularity (0 = full plane).
    pub fn plane_rows_for(&self,tokens:usize,page_bytes:usize)->usize {
        if self.ring {return self.rows;}
        let row_bytes=self.row_words.saturating_mul(4);
        if page_bytes==0 || row_bytes==0 || self.rows.saturating_mul(row_bytes)<page_bytes {
            return self.rows;
        }
        let page_rows=page_bytes.div_ceil(row_bytes).max(1);
        tokens.max(1).div_ceil(page_rows).saturating_mul(page_rows).min(self.rows)
    }
    /// Channels and rows of one convolution ring.
    pub fn ring_size(&self,ring:Ring)->(usize,usize) {
        match ring {Ring::Key|Ring::Value=>(self.kv_channels,self.kv_taps),Ring::AfterAttention|Ring::AfterExperts=>(self.channels,self.taps)}
    }
    /// The ring row of the input of the token at `position`.
    pub fn ring_slot(&self,ring:Ring,position:u64)->usize {(position%self.ring_size(ring).1 as u64) as usize}
    /// Words that one visit keeps: the two planes and the four rings.
    pub fn words(&self)->usize {2*self.rows*self.row_words+2*self.kv_taps*self.kv_channels+2*self.taps*self.channels}
}
/// What one visit keeps of a chat. `next` holds the rings after a step of many rows until that step is done
/// (`commit`), and `planes` the key and the value ring of a window layer after a step where the chat had more than
/// one row; outside such a step both are empty.
/// A step that keeps each row (`rows::Rows::keeping`) also leaves `every`, the ring after each of its rows
/// (`[rows][taps][channels]`), and `written`, the packed key and value rows of a window layer (`[rows][row words]`).
/// `Driver::accept` uses them and empties them.
pub struct VisitState<V> {pub keys:V,pub values:V,pub rings:[V;4],pub next:[Option<V>;4],pub planes:Option<[V;2]>,pub every:[Option<V>;4],pub written:Option<[V;2]>}
/// The state of a chat: the position of its next token, and one entry for each visit. The entry of a visit is at
/// the state slot that the description gives the visit (`Visit::slot`), not at its place in run order.
pub struct State<V> {pub position:u64,pub visits:Vec<VisitState<V>>}

/// One layout for each visit of the description, at the state slot of the visit, for steps of one row.
/// The slots must be the numbers 0 to the visit count less one, each once.
pub fn layouts(d:&LoopTextDescription,capacity:u64)->Result<Vec<Layout>> {layouts_for(d,capacity,Limits::ONE)}
/// The layouts for steps of up to `limits.step_rows` rows.
pub fn layouts_for(d:&LoopTextDescription,capacity:u64,limits:Limits)->Result<Vec<Layout>> {
    need(!d.visits.is_empty(),"the description lists no visit")?;
    let mut out=vec![None;d.visits.len()];
    for visit in &d.visits {
        let slot=out.get_mut(size(visit.slot)?).ok_or("loop decoder state: a visit names a state slot outside the visit count")?;
        need(slot.replace(Layout::for_steps(d,size(visit.layer)?,capacity,limits)?).is_none(),"two visits name one state slot")?;
    }
    out.into_iter().map(|layout|layout.ok_or_else(||"loop decoder state: a state slot has no visit".to_owned())).collect()
}
/// The state of a new chat: every word zero, position 0.
pub fn clean<C:Compute>(compute:&C,layouts:&[Layout])->Result<State<C::Value>> {
    let visits=layouts.iter().map(|l|Ok(VisitState {keys:compute.zeros(l.rows*l.row_words)?,values:compute.zeros(l.rows*l.row_words)?,
        rings:[compute.zeros(l.kv_taps*l.kv_channels)?,compute.zeros(l.kv_taps*l.kv_channels)?,compute.zeros(l.taps*l.channels)?,compute.zeros(l.taps*l.channels)?],
        next:std::array::from_fn(|_|None),planes:None,every:std::array::from_fn(|_|None),written:None})).collect::<Result<Vec<_>>>()?;
    Ok(State {position:0,visits})
}
/// Mapped rows of `plane`: fixed `layout.rows` for a ring, or `1..=layout.rows` whole rows for a growing straight plane.
pub fn plane_rows<C:Compute>(compute:&C,layout:&Layout,plane:&C::Value)->Result<usize> {
    let words=compute.words(plane);
    if layout.ring {
        need(words==layout.rows*layout.row_words,"a ring plane differs from the layout")?;
        return Ok(layout.rows);
    }
    need(words>0 && words.is_multiple_of(layout.row_words),"a straight plane differs from the row width")?;
    let rows=words/layout.row_words;
    need(rows<=layout.rows,"a straight plane exceeds the layout capacity")?;
    Ok(rows)
}
/// Ensure every straight KV plane of `state` is mapped for at least `needed_tokens` tokens (capped at `layout.rows`).
/// Call outside `compute.work` so CUDA VMM page maps never execute inside an unsubmitted resident launch queue.
pub fn ensure_capacity<C:Compute>(compute:&C,layouts:&[Layout],state:&mut State<C::Value>,needed_tokens:usize)->Result<()> {
    need(state.visits.len()==layouts.len(),"the state differs from the visits of the description")?;
    let page_bytes=compute.plane_page_bytes();
    for (visit,layout) in state.visits.iter_mut().zip(layouts) {
        if layout.ring {continue;}
        let want_words=layout.plane_rows_for(needed_tokens,page_bytes)*layout.row_words;
        let max_words=layout.rows*layout.row_words;
        if compute.words(&visit.keys)<want_words {compute.plane_grow(&mut visit.keys,want_words,max_words)?;}
        if compute.words(&visit.values)<want_words {compute.plane_grow(&mut visit.values,want_words,max_words)?;}
    }
    Ok(())
}
/// Live global KV occupies only the committed rows. Wrapped windows must keep their original slot layout.
fn prefix_plane_words(layout:&Layout,position:u64)->usize {
    layout.row_words*if layout.ring {layout.rows} else {(position as usize).min(layout.rows)}
}
/// Private (non-straight-plane) bytes retained by one snapshot: window rings, convolution rings, and metadata.
pub(super) fn retained_private_bytes<V>(layouts:&[Layout])->usize {
    let bytes=|words:usize|(words*4).next_multiple_of(4096);
    layouts.iter().map(|l| {
        let rings=2*(bytes(l.kv_taps*l.kv_channels)+bytes(l.taps*l.channels));
        let window=if l.ring {2*bytes(l.rows*l.row_words)} else {0};
        window+rings+std::mem::size_of::<VisitState<V>>()
    }).sum()
}
pub(super) fn prefix_bytes<V>(layouts:&[Layout],position:u64,page_bytes:usize)->usize {
    let bytes=|words:usize|(words*4).next_multiple_of(4096);
    layouts.iter().map(|l| {
        let plane=if l.ring {
            bytes(l.rows*l.row_words)
        } else {
            let rows=l.plane_rows_for(position as usize,page_bytes);
            bytes(rows*l.row_words)
        };
        2*(plane+bytes(l.kv_taps*l.kv_channels)+bytes(l.taps*l.channels))
            +std::mem::size_of::<VisitState<V>>()
    }).sum()
}
/// Copy a committed state at its exact position inside an active unrecorded `compute.work` scope.
/// Straight planes use zero-copy CoW page aliasing (`plane_fork`) up to complete pages and copy only the
/// partial tail; `restore` maps writable capacity for `target_tokens`.
pub(super) fn copy_prefix_unrecorded<C:Compute>(compute:&C,layouts:&[Layout],state:&State<C::Value>,restore:bool,target_tokens:usize)->Result<State<C::Value>> {
    need(state.position>0 && state.visits.len()==layouts.len(),"invalid prefix state")?;
    let page_bytes=compute.plane_page_bytes();
    let visits=state.visits.iter().zip(layouts).map(|(visit,layout)| {
        need(visit.planes.is_none() && visit.next.iter().all(Option::is_none) && visit.every.iter().all(Option::is_none) && visit.written.is_none(),"cannot copy an unfinished step")?;
        let copy=|source:&C::Value| compute.rows_append(None,0,source,1);
        let plane=|source:&C::Value| {
            if layout.ring {
                let words=prefix_plane_words(layout,state.position);
                let live=compute.range(source,0,words)?;
                return copy(&live);
            }
            let committed=prefix_plane_words(layout,state.position);
            let max_words=layout.rows*layout.row_words;
            let want_tokens=if restore {target_tokens.max(state.position as usize)} else {state.position as usize};
            let needed=layout.plane_rows_for(want_tokens,page_bytes)*layout.row_words;
            compute.plane_fork(source,committed,needed,max_words)
        };
        let [k,v,attention,experts]=&visit.rings;
        Ok(VisitState {keys:plane(&visit.keys)?,values:plane(&visit.values)?,
            rings:[copy(k)?,copy(v)?,copy(attention)?,copy(experts)?],
            next:std::array::from_fn(|_|None),planes:None,every:std::array::from_fn(|_|None),written:None})
    }).collect::<Result<Vec<_>>>()?;
    Ok(State {position:state.position,visits})
}
/// Numbers for an arm, one 32-bit word each: a live count, a slot.
pub fn step_words<C:Compute>(compute:&C,numbers:&[usize])->Result<C::Value> {
    need(numbers.iter().all(|&n|n<=STEP_WORD_LIMIT),"a count or a slot is too large for a step word")?;
    compute.upload_constant(&numbers.iter().map(|&n|n as f32).collect::<Vec<_>>())
}
/// Two BF16 values a word. The values must be BF16 values already: packing keeps their bits.
pub fn pack<C:Compute>(compute:&C,values:&C::Value)->Result<C::Value> {
    let words=compute.words(values);need(words>0 && words.is_multiple_of(2),"a packed row needs an even count of values")?;
    let shape=Shape {op:Op::PackBf16,heads:0,dim:0,taps:0,rows:u32::try_from(words/2).map_err(|_|"loop decoder state: a packed row exceeds the launch format")?,cols:0,epsilon:0.0,bound:0.0};
    compute.operation(shape,[Some(values),None,None,None],&[])
}
/// Put the finished key or value of the token at `position` into its row of `plane`. `finished` holds the BF16
/// values of one row, not packed. No other row changes.
pub fn write_row<C:Compute>(compute:&C,layout:&Layout,plane:&mut C::Value,position:u64,finished:&C::Value)->Result<()> {
    let cap=plane_rows(compute,layout,plane)?;
    let slot=layout.slot(position)?;
    need(compute.words(finished)==layout.kv_channels && slot<cap,"a key or value row differs from the layout")?;
    let packed=pack(compute,finished)?;compute.state_fill(plane,slot*layout.row_words,&packed)
}
/// Put the input of one convolution for the token at `position` into its ring row. No other row changes.
/// It gives the step word of arm `ShortConv`: the ring row of the current token.
pub fn write_ring<C:Compute>(compute:&C,layout:&Layout,which:Ring,rings:&mut [C::Value;4],position:u64,input:&C::Value)->Result<C::Value> {
    let ((channels,taps),slot,ring)=(layout.ring_size(which),layout.ring_slot(which,position),&mut rings[which as usize]);
    need(compute.words(input)==channels && compute.words(ring)==taps*channels,"a convolution input differs from the layout")?;
    compute.state_fill(ring,slot*channels,input)?;step_words(compute,&[slot])
}
/// The first `rows` rows of a plane, for the attention arms: `rows` is `Layout::view_rows` of the token.
/// The two step words of those arms are the live count and the row of the token (`Layout::live`, `Layout::slot`).
pub fn view<C:Compute>(compute:&C,layout:&Layout,plane:&C::Value,rows:usize)->Result<C::Value> {
    let cap=plane_rows(compute,layout,plane)?;
    need(rows>0 && rows<=cap,"a plane or its view differs from the layout")?;
    compute.range(plane,0,rows*layout.row_words)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bounded_decoder::Host,loop_decoder::fixture::Fixture};
    fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
    /// BF16 values of one row: each word is `base + index` in its high half.
    fn row(words:usize,base:u32)->Vec<f32> {(0..words as u32).map(|i|f32::from_bits((0x3f00+base+i)<<16)).collect()}
    #[test]
    fn the_layouts_come_from_the_description_one_a_visit() {
        let fixture=Fixture::small();let d=&fixture.description;let l=layouts(d,16).unwrap();
        assert_eq!(l.len(),d.visits.len());
        // The layouts stand at the state slot of each visit. In the wide shape the slots are not the run order:
        // the layers after the loop keep the slots of their layers, and the later passes come after every layer.
        let wide_description=Fixture::wide().description;
        assert_eq!(wide_description.visits.iter().map(|visit|visit.slot).collect::<Vec<_>>(),[0,1,2,3,4,5,8,9,10,11,12,13,14,15,6,7]);
        let mut twice=d.clone();twice.visits[5].slot=4;assert!(layouts(&twice,16).unwrap_err().contains("two visits name one state slot"));
        let mut outside=d.clone();outside.visits[5].slot=6;assert!(layouts(&outside,16).unwrap_err().contains("outside the visit count"));
        let mut none=d.clone();none.visits.clear();assert!(layouts(&none,16).unwrap_err().contains("lists no visit"));
        // Visit 0 is a window layer: a ring of the window's rows. Visit 1 sees every token: a plane of the capacity.
        assert_eq!(l[0],Layout {rows:4,window:4,step_rows:1,key_bytes:1<<30,ring:true,row_words:8,kv_channels:16,kv_taps:4,channels:32,taps:4});
        assert_eq!(l[1],Layout {rows:16,window:16,step_rows:1,key_bytes:1<<30,ring:false,row_words:4,kv_channels:8,kv_taps:4,channels:32,taps:4});
        // For steps of up to 5 rows the state ring stays the window, the wide ring has 4 more rows and a token still sees the window. A plane does not change.
        let wider=layouts_for(d,16,Limits {step_rows:5,key_bytes:64}).unwrap();
        assert_eq!((wider[0].rows,wider[0].window,wider[0].live(2).unwrap(),wider[0].live(9).unwrap(),wider[0].slot(9).unwrap(),wider[0].view_rows(9).unwrap()),(4,4,3,4,1,4));
        // The state keeps a window ring; a chat with many rows in a step works in the wider ring.
        let wide=wider[0].wide();assert_eq!((wide.rows,wide.slot(9).unwrap(),wide.view_rows(9).unwrap(),wide.live(9).unwrap()),(8,1,8,4));
        assert_eq!(Layout {step_rows:1,key_bytes:1<<30,..wider[0]},l[0]);assert_eq!(l[0].wide(),l[0]);
        assert_eq!(Layout {step_rows:1,key_bytes:1<<30,..wider[1]},l[1]);assert_eq!(layouts_for(d,16,Limits::ONE).unwrap(),l);
        // A step row limit of one whole check pass or more: the state ring keeps the spare rows of one check pass.
        // A token still sees the window, its slot goes round the larger ring, and a plane does not change.
        assert_eq!((spare_rows(1),spare_rows(CHECK_ROWS-1),spare_rows(CHECK_ROWS),spare_rows(4096)),(0,0,CHECK_ROWS-1,CHECK_ROWS-1));
        for limit in [CHECK_ROWS,CHECK_ROWS+5] {
            let spare=layouts_for(d,16,Limits {step_rows:limit,key_bytes:64}).unwrap();let ring=4+CHECK_ROWS-1;
            assert_eq!((spare[0].rows,spare[0].window,spare[0].wide().rows,spare[0].view_rows(40).unwrap()),(ring,4,4+limit-1,ring));
            assert_eq!((spare[0].live(2).unwrap(),spare[0].live(40).unwrap(),spare[0].slot(40).unwrap(),spare[0].slot(ring as u64).unwrap()),(3,4,40%ring,0));
            assert_eq!(Layout {rows:4,..spare[0]},Layout {step_rows:limit,key_bytes:64,..l[0]});assert_eq!(Layout {step_rows:1,key_bytes:1<<30,..spare[1]},l[1]);
            assert_eq!(spare[0].words(),2*ring*8+2*4*16+2*4*32);
        }
        assert_eq!(layouts_for(d,16,Limits {step_rows:CHECK_ROWS-1,key_bytes:64}).unwrap()[0].rows,4);
        assert!(layouts_for(d,16,Limits {step_rows:0,key_bytes:64}).is_err() && layouts_for(d,16,Limits {step_rows:2,key_bytes:0}).is_err());
        // The two visits of one layer have the same layout and their own state.
        assert_eq!((l[2],l[3]),(l[4],l[5]));assert_eq!(l[0].words(),2*4*8+2*4*16+2*4*32);
        let wide=layouts(&wide_description,12).unwrap();
        assert_eq!(wide.len(),16);assert_eq!((wide[6],wide[7],wide[8]),(wide[0],wide[1],wide[2]),"slots 6 and 7 are layers 6 and 7; slot 8 is layer 2 in pass 2");assert_eq!((wide[0].rows,wide[0].kv_taps,wide[0].taps,wide[0].row_words),(3,3,3,36));assert_eq!((wide[1].rows,wide[1].ring,wide[1].row_words),(12,false,40));
        assert!(layouts(d,0).is_err() && layouts(d,d.max_positions+1).is_err());
        assert!(Layout::new(d,4,16).unwrap_err().contains("outside the description"));
    }
    #[test]
    fn the_slot_is_a_function_of_the_position_alone() {
        let l=layouts(&Fixture::small().description,16).unwrap();let (ring,plane)=(l[0],l[1]);
        assert_eq!((0..10).map(|p|ring.slot(p).unwrap()).collect::<Vec<_>>(),[0,1,2,3,0,1,2,3,0,1]);
        assert_eq!((0..10).map(|p|ring.live(p).unwrap()).collect::<Vec<_>>(),[1,2,3,4,4,4,4,4,4,4]);
        assert_eq!((0..10).map(|p|plane.slot(p).unwrap()).collect::<Vec<_>>(),(0..10).collect::<Vec<_>>());
        assert_eq!((0..10).map(|p|plane.live(p).unwrap()).collect::<Vec<_>>(),(1..11).collect::<Vec<_>>());
        // A straight plane ends at its capacity. A ring has no end.
        assert!(plane.slot(15).is_ok() && plane.slot(16).unwrap_err().contains("longer than the state's capacity") && plane.live(16).is_err());
        assert_eq!((ring.slot(u64::MAX).unwrap(),ring.live(u64::MAX).unwrap()),(3,4));
        // In a straight plane the row of the current token is the live count less one.
        for p in 0..16 {assert_eq!(plane.slot(p).unwrap()+1,plane.live(p).unwrap());}
        assert_eq!((0..9).map(|p|ring.ring_slot(Ring::Key,p)).collect::<Vec<_>>(),[0,1,2,3,0,1,2,3,0]);
        let wide=layouts(&Fixture::wide().description,12).unwrap()[0];
        assert_eq!((0..7).map(|p|wide.ring_slot(Ring::AfterExperts,p)).collect::<Vec<_>>(),[0,1,2,0,1,2,0]);
    }
    #[test]
    fn a_ring_is_read_whole_and_a_plane_in_buckets() {
        let l=layouts(&Fixture::small().description,16).unwrap();
        assert_eq!((l[0].view_rows(0).unwrap(),l[0].view_rows(9).unwrap()),(4,4));
        // A small plane is below one bucket, so the view is the whole plane.
        assert_eq!((l[1].view_rows(0).unwrap(),l[1].view_rows(15).unwrap()),(16,16));
        let bucket=BUCKET_ROWS as usize;let plane=Layout {rows:2*bucket+100,ring:false,..l[1]};
        assert_eq!([0,bucket as u64-1,bucket as u64,2*bucket as u64-1,2*bucket as u64,2*bucket as u64+99].map(|p|plane.view_rows(p).unwrap()),[bucket,bucket,2*bucket,2*bucket,2*bucket+100,2*bucket+100]);
        assert!(plane.view_rows(2*bucket as u64+100).is_err());
    }
    #[test]
    fn a_new_chat_is_all_zero() {
        let l=layouts(&Fixture::small().description,16).unwrap();let state=clean(&Host,&l).unwrap();
        assert_eq!((state.position,state.visits.len()),(0,6));
        for (visit,layout) in state.visits.iter().zip(&l) {
            assert_eq!((visit.keys.len(),visit.values.len()),(layout.rows*layout.row_words,layout.rows*layout.row_words));
            assert_eq!(visit.rings.iter().map(Vec::len).collect::<Vec<_>>(),[layout.kv_taps*layout.kv_channels,layout.kv_taps*layout.kv_channels,layout.taps*layout.channels,layout.taps*layout.channels]);
            assert_eq!(visit.keys.len()*2+visit.rings.iter().map(Vec::len).sum::<usize>(),layout.words());
            assert!(visit.keys.iter().chain(&visit.values).chain(visit.rings.iter().flatten()).all(|v|v.to_bits()==0));
        }
    }
    #[test]
    fn a_row_goes_to_its_slot_packed_and_no_other_word_changes() {
        let l=layouts(&Fixture::small().description,16).unwrap();
        for layout in [l[0],l[1]] {
            let mut state=clean(&Host,&[layout]).unwrap();let plane=&mut state.visits[0].keys;let mut expected=vec![0u32;layout.rows*layout.row_words];
            for position in 0..11u64 {
                let finished=row(layout.kv_channels,position as u32*64);write_row(&Host,&layout,plane,position,&finished).unwrap();
                let slot=if layout.ring {position as usize%layout.rows} else {position as usize};
                // Value 2i is the low half of word i and value 2i + 1 is its high half.
                for i in 0..layout.row_words {expected[slot*layout.row_words+i]=(finished[2*i].to_bits()>>16)|(finished[2*i+1].to_bits()&0xffff_0000);}
                assert_eq!(bits(plane),expected,"position {position}");
            }
            // After 11 tokens a ring of 4 rows holds positions 8, 9, 10 and 7, each in the slot of its position.
            if layout.ring {for position in 7..11u64 {assert_eq!((plane[(position as usize%4)*layout.row_words].to_bits()&0xffff)<<16,row(1,position as u32*64)[0].to_bits());}}
        }
        let mut plane=clean(&Host,&[l[1]]).unwrap().visits.remove(0).keys;
        assert!(write_row(&Host,&l[1],&mut plane,16,&row(l[1].kv_channels,0)).is_err(),"a plane has no row past its capacity");
        assert!(write_row(&Host,&l[1],&mut plane,0,&row(l[1].kv_channels+2,0)).unwrap_err().contains("differs from the layout"));
        assert!(write_row(&Host,&l[0],&mut plane,0,&row(l[0].kv_channels,0)).is_err(),"a plane of another layout is refused");
        assert!(plane.iter().all(|v|v.to_bits()==0),"a refused write changes no word");
    }
    #[test]
    fn a_convolution_input_goes_to_the_ring_row_of_its_position() {
        let l=layouts(&Fixture::wide().description,12).unwrap()[0];let mut state=clean(&Host,&[l]).unwrap();let rings=&mut state.visits[0].rings;
        for which in Ring::ALL {
            let (channels,taps)=l.ring_size(which);let mut expected=vec![0u32;taps*channels];
            for position in 0..8u64 {
                // A ring holds the 32-bit words of its input as they are: here words that are no BF16 values.
                let input=(0..channels).map(|i|1.0+(position as usize*channels+i) as f32/65536.0).collect::<Vec<_>>();
                let step=write_ring(&Host,&l,which,rings,position,&input).unwrap();let slot=position as usize%taps;
                assert_eq!(step,[slot as f32]);expected[slot*channels..(slot+1)*channels].copy_from_slice(&bits(&input));
                assert_eq!(bits(&rings[which as usize]),expected,"{which:?} position {position}");
            }
        }
        assert_eq!((l.ring_size(Ring::Key),l.ring_size(Ring::AfterExperts)),((72,3),(160,3)));
        assert!(write_ring(&Host,&l,Ring::Key,rings,0,&vec![0.0;160]).unwrap_err().contains("differs from the layout"));
    }
    #[test]
    fn the_view_gives_the_rows_the_live_count_and_the_slot() {
        let l=layouts(&Fixture::small().description,16).unwrap();
        for (layout,position,live,slot) in [(l[0],0u64,1usize,0usize),(l[0],3,4,3),(l[0],6,4,2),(l[1],0,1,0),(l[1],6,7,6)] {
            let mut plane=Host.zeros(layout.rows*layout.row_words).unwrap();
            write_row(&Host,&layout,&mut plane,position,&row(layout.kv_channels,7)).unwrap();
            let count=layout.view_rows(position).unwrap();let rows=view(&Host,&layout,&plane,count).unwrap();
            assert_eq!((count,layout.live(position).unwrap(),layout.slot(position).unwrap()),(layout.rows,live,slot));assert_eq!(bits(&rows),bits(&plane));
            assert_eq!(bits(&view(&Host,&layout,&plane,1).unwrap()),bits(&plane[..layout.row_words]));
            assert!(view(&Host,&layout,&plane,0).is_err() && view(&Host,&layout,&plane,layout.rows+1).is_err());
        }
        assert!(view(&Host,&l[0],&Host.zeros(3).unwrap(),1).is_err());
        assert_eq!(step_words(&Host,&[STEP_WORD_LIMIT]).unwrap(),[16_777_216.0]);
        assert!(step_words(&Host,&[STEP_WORD_LIMIT+1]).unwrap_err().contains("too large for a step word"));
    }
}
