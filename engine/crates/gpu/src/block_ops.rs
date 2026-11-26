//! The block arms of Inkling 2 Mini: safe shared code, one source for the host and the GPU. On the GPU they run in
//! the Inkling image only (crate `kernels-inkling`, entries `resident_launch_block_arms` and `resident_launch_block_rows`).
//! The arms live in three files. Each file gives the `lengths` and the `unit` of its own arms.
//! An arm that has no body yet has no lengths, so the host refuses its launch.
//!
//! Rules of every arm:
//! - It reads and writes only the slices it is given. An index that comes from data is tested before use.
//! - A product and its sum are two rounded steps. `exp`, `sigmoid`, `tanh` and the BF16 round come from `fixed_math`.
//! - A size comes from the shape. A constant of the operation is a named constant in the arm's file.
#[cfg(not(target_arch="nvptx64"))]
pub use crate::causal_ops::{Op,Shape};
#[cfg(target_arch="nvptx64")]
pub use device::{Op,Shape};
/// The arms as the GPU reads them: their own names, shape and launch record, with the numbers and the layout of the
/// host's (`causal_ops::Op`, `Shape`, `Arguments`; host test). The general kernels keep the list of private main; only
/// the Inkling image (`kernels-inkling`) reads these.
pub mod device {
    /// The number of the first arm: the count of the older operations of `causal_ops::Op`.
    pub const FIRST:u32=45;
    #[repr(u32)]
    #[derive(Debug,Clone,Copy,PartialEq,Eq)]
    pub enum Op {HeadClip=FIRST,RelBias,ShortConv,WindowScores,SinkSoftmax,WindowValues,MixBlend,GateBlend,CappedGate,RouteFold,Softcap,PlainNorm,RoundedNorm,RowDots,RouterPick}
    #[repr(C)]
    #[derive(Debug,Clone,Copy)]
    pub struct Shape {pub op:Op,pub heads:u32,pub dim:u32,pub taps:u32,pub rows:u32,pub cols:u32,pub epsilon:f32,pub bound:f32}
    #[repr(C)]
    #[derive(Clone,Copy)]
    pub struct Arguments {pub input:[u64;4],pub weight:u64,pub output:u64,pub shape:Shape}
    /// A launch of many rows of one arm (`block_rows`, entry `resident_launch_block_rows`): the arguments of the arm,
    /// then the row count and the mask of shared inputs.
    #[repr(C)]
    #[derive(Clone,Copy)]
    pub struct RowsArguments {pub arguments:Arguments,pub rows:u32,pub shared:u32}
    /// Where the row count and the mask lie in the payload of a recorded rows launch.
    pub const ROWS_AT:usize=core::mem::offset_of!(RowsArguments,rows);
    pub const SHARED_AT:usize=core::mem::offset_of!(RowsArguments,shared);
    /// The number of the last arm. The entry reads the op word of a command as a `u32` at `OP_AT` and takes the
    /// command as `Arguments` only when that word is in `FIRST..=LAST` (an `Op` of any other value is no valid value).
    pub const LAST:u32=Op::RouterPick as u32;
    /// Where the op word lies in `Arguments`, and so in `RowsArguments` (its first field).
    pub const OP_AT:usize=core::mem::offset_of!(Arguments,shape)+core::mem::offset_of!(Shape,op);
}
/// Word counts of one launch: the 4 inputs, the weight, the output units, and the words of one unit.
/// The square root, correctly rounded: on the GPU the one instruction of the Inkling image (`inkling_math::root`,
/// `sqrt.rn.f32`, the same instruction as the general image's), on the host `f32::sqrt`. No GLM file is imported.
#[inline(always)]
pub(crate) fn root(x:f32)->f32 {
    #[cfg(target_arch="nvptx64")] {crate::inkling_math::root(x)}
    #[cfg(not(target_arch="nvptx64"))] {x.sqrt()}
}
pub type Lengths=([usize;4],usize,usize,usize);
/// Lanes of a fixed sum tree. A part of the sum order, not a size of a model.
pub const LANES:usize=32;
/// Values in one chunk of the norm tree.
pub const NORM_CHUNK:usize=128;
/// Join the lane sums: for the distances 16, 8, 4, 2, 1, each lane below the distance adds its upper partner.
#[inline(always)]
pub fn join(mut lanes:[f32;LANES])->f32 {
    for distance in [16,8,4,2,1] {for lane in 0..distance {lanes[lane]+=lanes[lane+distance];}}
    lanes[0]
}
/// The sum of squares in the tree of the norm kernel (`model_rms_partials`, `model_rms_apply`): chunks of 128 values;
/// in a chunk, lane = index mod 32, each lane a chain in rising index from +0.0, the lanes joined; then the chunk
/// sums with lane = chunk mod 32, each lane a chain in rising chunk number from +0.0, the lanes joined.
#[inline(always)]
pub fn norm_square_sum(x:&[f32])->f32 {
    let mut outer=[0.0f32;LANES];
    let (mut chunk,mut begin)=(0,0);
    while begin<x.len() {
        let end=if x.len()-begin>NORM_CHUNK {begin+NORM_CHUNK} else {x.len()};
        let mut lanes=[0.0f32;LANES];
        for (lane,sum) in lanes.iter_mut().enumerate() {let mut i=begin+lane;while i<end {*sum+=x[i]*x[i];i+=LANES;}}
        outer[chunk%LANES]+=join(lanes);
        chunk+=1;begin=end;
    }
    join(outer)
}
/// Words of one unit of a norm arm (`PlainNorm`, `RoundedNorm`) when the width is a multiple of it. A part of the
/// launch layout, not a size of a model: a row is then written by `width / NORM_PART` GPU threads, and each of them
/// makes the sum of the squares of the whole row itself, in the one fixed order. So the words do not depend on it.
pub const NORM_PART:usize=32;
/// The words of one unit of a norm arm of width `d`: `NORM_PART`, or the whole row when the width is no multiple.
#[inline(always)]
pub fn norm_part(d:usize)->usize {if d.is_multiple_of(NORM_PART) {NORM_PART} else {d}}
/// `index / size` and `index mod size` for the index of a unit and a size of its shape. Both are below 2^32
/// (`Shape::lengths` refuses a launch with more words), and a GPU thread divides 32-bit numbers much faster than
/// 64-bit ones.
#[inline(always)]
pub fn split(index:usize,size:usize)->(usize,usize) {let (i,n)=(index as u32,size as u32);((i/n) as usize,(i%n) as usize)}
/// The terms of a fixed sum: `term(at)` is the term of place `at`.
pub trait Terms {fn term(&self,at:usize)->f32;}
/// Lanes that are 8 apart: the lanes `l`, `l + 8`, `l + 16` and `l + 24` are one part of the join.
pub(crate) const APART:usize=8;
/// The join of the 8 parts: part `l` is the joined sum of the lanes `l`, `l + 8`, `l + 16`, `l + 24`. What is left
/// of `join` is its last three distances: 4, 2 and 1.
#[inline(always)]
pub(crate) fn join_parts(parts:[f32;APART])->f32 {((parts[0]+parts[4])+(parts[2]+parts[6]))+((parts[1]+parts[5])+(parts[3]+parts[7]))}
/// One part of the sum of `lane_sum`: the lanes `lane`, `lane + 8`, `lane + 16` and `lane + 24`, each a chain in
/// rising place from +0.0, joined as `join` joins them (distance 16, then distance 8).
/// A function of its own on purpose, with four sums and one loop: on the GPU its partial sums do not add to the
/// registers of the thread that calls it. The registers of one thread set how many threads a block of the entry may
/// hold (`kernels-inkling/REGISTERS.txt`).
#[inline]
pub(crate) fn lane_part<T:Terms>(terms:&T,first:usize,end:usize,lane:usize)->f32 {
    let (mut a,mut b,mut c,mut d)=(0.0f32,0.0f32,0.0f32,0.0f32);let mut base=first+lane;
    while base+3*APART<end {a+=terms.term(base);b+=terms.term(base+APART);c+=terms.term(base+2*APART);d+=terms.term(base+3*APART);base+=LANES;}
    // The last places, fewer than a row of 32: a lane with no place left gets nothing.
    if base<end {a+=terms.term(base);}
    if base+APART<end {b+=terms.term(base+APART);}
    if base+2*APART<end {c+=terms.term(base+2*APART);}
    (a+c)+(b+d)
}
/// The sum of `terms.term(at)` for the places `first` to `end` on the 32-lane tree: the term of place `at` goes to
/// lane `(at - first) mod LANES`, each lane is a chain in rising place from +0.0, and the lanes are joined as in
/// `join`. The lanes are taken four at a time, so each lane gets its terms in the order of its chain.
#[inline(always)]
pub fn lane_sum<T:Terms>(terms:&T,first:usize,end:usize)->f32 {
    let mut parts=[0.0f32;APART];
    for (lane,part) in parts.iter_mut().enumerate() {*part=lane_part(terms,first,end,lane);}
    join_parts(parts)
}
/// `causal_ops::square_sum` for the arms: the same lane chains and the same join, so the same bits for every row
/// (tested on the host at every width class). The row is read in rows of 32 values, four lanes a loop, as
/// `lane_part` adds them. A function of its own on purpose, as `lane_part`. Four sums and not more: with eight the
/// kernel that held the arms then used 102 registers a thread against 95 (measured, before the Inkling image).
#[inline]
pub fn row_squares(x:&[f32])->f32 {
    let (rows,rest)=x.as_chunks::<LANES>();
    let mut parts=[0.0f32;APART];
    for (lane,part) in parts.iter_mut().enumerate() {
        let (mut a,mut b,mut c,mut d)=(0.0f32,0.0f32,0.0f32,0.0f32);
        for row in rows {
            a+=row[lane]*row[lane];b+=row[lane+APART]*row[lane+APART];c+=row[lane+2*APART]*row[lane+2*APART];d+=row[lane+3*APART]*row[lane+3*APART];
        }
        // The last values, fewer than 32: a lane with no value left gets nothing.
        if lane<rest.len() {a+=rest[lane]*rest[lane];}
        if lane+APART<rest.len() {b+=rest[lane+APART]*rest[lane+APART];}
        if lane+2*APART<rest.len() {c+=rest[lane+2*APART]*rest[lane+2*APART];}
        if lane+3*APART<rest.len() {d+=rest[lane+3*APART]*rest[lane+3*APART];}
        *part=(a+c)+(b+d);
    }
    join_parts(parts)
}
/// The chain of lane `lane` of one full chunk of the norm tree: 4 squares in rising index from +0.0.
#[inline(always)]
fn chunk_lane(chunk:&[f32;NORM_CHUNK],lane:usize)->f32 {
    let mut sum=0.0f32;
    sum+=chunk[lane]*chunk[lane];sum+=chunk[lane+LANES]*chunk[lane+LANES];sum+=chunk[lane+2*LANES]*chunk[lane+2*LANES];sum+=chunk[lane+3*LANES]*chunk[lane+3*LANES];
    sum
}
/// The sum of one full chunk of the norm tree: its 32 lane chains, joined, in 8 parts of 4 lanes.
#[inline(always)]
fn chunk_sum(chunk:&[f32;NORM_CHUNK])->f32 {
    let mut parts=[0.0f32;APART];
    for (lane,part) in parts.iter_mut().enumerate() {
        *part=(chunk_lane(chunk,lane)+chunk_lane(chunk,lane+2*APART))+(chunk_lane(chunk,lane+APART)+chunk_lane(chunk,lane+3*APART));
    }
    join_parts(parts)
}
/// `norm_square_sum` for the arms: the same sums in the same order, so the same bits for every row (tested on the
/// host at every width class). A full chunk is added in 8 parts of 4 lanes; a last chunk that is not full uses the
/// definition. A function of its own on purpose, as `lane_part`.
#[inline]
pub fn norm_squares(x:&[f32])->f32 {
    let (chunks,rest)=x.as_chunks::<NORM_CHUNK>();
    let mut outer=[0.0f32;LANES];
    for (index,chunk) in chunks.iter().enumerate() {
        outer[index%LANES]+=chunk_sum(chunk);
    }
    // A last chunk that is not full: `norm_square_sum` of one chunk alone is the join of its lanes, and then that sum
    // plus zeros, which changes no bit of a sum of squares.
    if !rest.is_empty() {outer[chunks.len()%LANES]+=norm_square_sum(rest);}
    join(outer)
}
/// Threads of one block of a launch whose rows share a sum (`shared_sum`): a part of the launch layout.
pub const SUM_THREADS:usize=128;
/// Rows of a shared-sum launch from which one block makes a whole row (`shared_span`): a part of the launch layout,
/// not a size of a model. With fewer rows a block makes `SUM_THREADS` words of a row, so a short launch still has
/// many blocks; with this many rows there are blocks enough, and a row's parts are made once, not once a block.
pub const WHOLE_ROWS:usize=64;
/// The words of a row that one block of a shared-sum launch writes: the whole row from `WHOLE_ROWS` rows on, else
/// `SUM_THREADS`. Either way a multiple of `SUM_THREADS` that divides the width.
#[inline(always)]
pub fn shared_span(s:Shape)->usize {if s.heads as usize>=WHOLE_ROWS {s.dim as usize} else {SUM_THREADS}}
/// A launch whose rows each share ONE sum (`PlainNorm`, `RoundedNorm`, `HeadClip` at a width that is a multiple of
/// `SUM_THREADS`): (blocks, parts of the sum). On the GPU a block of `SUM_THREADS` threads then writes
/// `shared_span` words of one row, thread `t` the `span / SUM_THREADS` words from `t * span / SUM_THREADS` on (one
/// call of `shared_words`, so a thread holds no more state than with one word): thread `p` makes
/// part `p` of the sum of the row (`sum_part`), the threads of the block meet, and each thread joins the parts and
/// writes its words (`shared_words`). The parts and their join are the very sums of `unit`, in the same order, so
/// the words are those of `unit` with either span (host test).
#[inline(always)]
pub fn shared_sum(s:Shape)->Option<(usize,usize)> {
    let (h,d)=(s.heads as usize,s.dim as usize);
    if !matches!(s.op,Op::PlainNorm|Op::RoundedNorm|Op::HeadClip) || d==0 || !d.is_multiple_of(SUM_THREADS) {return None;}
    let parts=if s.op==Op::RoundedNorm {d/NORM_CHUNK} else {LANES};
    if parts>LANES {return None;}
    Some((h.checked_mul(d/shared_span(s))?,parts))
}
/// Words of each of the two rows a staged `RowDots` block holds in shared memory at a time (a tile).
pub const DOT_TILE:usize=512;
/// The most dots a staged `RowDots` launch takes: one block a dot pays only while the dots are few.
pub const DOT_STAGE_DOTS:usize=512;
/// `RowDots` staged (`Some((dots, width))`): a block of `SUM_THREADS` threads a dot. The threads copy the dot's input
/// row and weight row into shared memory a tile at a time, all at once, and one thread runs the dot's chain from there
/// (`block_stream::dot_steps`, the steps of the unit). The chain and its order are those of the unit, so the words are.
#[inline]
pub fn dot_stage(s:Shape)->Option<(usize,usize)> {
    if s.op!=Op::RowDots {return None;}
    let (_,_,dots,_)=checked_lengths(s)?;
    (dots<=DOT_STAGE_DOTS).then_some((dots,s.cols as usize))
}
/// `HeadClip` with heads of at most `LANES` values (`Some(blocks)`): a warp a head, `SUM_THREADS / LANES` heads a
/// block of `SUM_THREADS` threads. Lane `l` makes the chain of lane `l` of the squares of its head
/// (`block_attention::clip_lane`: the one term at place `l`, or none), the warp joins the 32 lane chains in the order
/// of `join`, and lane `l < d` writes word `l` of the head (`block_attention::clipped`). With `d <= LANES` those are
/// the chains and the join of `unit` (`lane_sum` over the `d` places), so the words are.
#[inline]
pub fn clip_warps(s:Shape)->Option<usize> {
    let (h,d)=(s.heads as usize,s.dim as usize);
    if s.op!=Op::HeadClip || d==0 || d>LANES {return None;}
    checked_lengths(s)?;
    Some(h.div_ceil(SUM_THREADS/LANES))
}
/// Launch geometry (blocks, threads a block) of an arm: a block a dot for a staged `RowDots` (`dot_stage`); a block a
/// head for a staged `SinkSoftmax` (`block_attention::softmax_stage`); a block for each `VALUES_CHANNELS` words of a
/// staged `WindowValues` (`block_attention::values_stage`); a block for each `shared_span` words of a row when the
/// rows share a sum (`shared_sum`: `SUM_THREADS` words, or a whole row from `WHOLE_ROWS` rows on); else one thread a
/// unit in blocks of 128.
#[inline]
pub fn geometry(s:Shape,units:usize)->(u32,u32) {
    if let Some((dots,_))=dot_stage(s) {return (dots as u32,SUM_THREADS as u32);}
    if let Some(Ok(blocks))=clip_warps(s).map(u32::try_from) {return (blocks,SUM_THREADS as u32);}
    if let Some(Ok(blocks))=crate::block_attention::softmax_stage(s).or_else(||crate::block_attention::values_stage(s)).map(u32::try_from) {return (blocks,SUM_THREADS as u32);}
    match shared_sum(s) {Some((blocks,_)) if u32::try_from(blocks).is_ok()=>(blocks as u32,SUM_THREADS as u32),_=>(units.div_ceil(128) as u32,128)}
}
/// The chain of lane `lane` of the squares of one row: rising index from +0.0 (a lane of `row_squares`).
#[inline]
pub fn lane_squares(row:&[f32],lane:usize)->f32 {let mut sum=0.0f32;let mut i=lane;while i<row.len() {sum+=row[i]*row[i];i+=LANES;}sum}
/// The sum of chunk `chunk` of the norm tree of one row (a chunk sum of `norm_squares`).
#[inline]
pub fn norm_chunk(row:&[f32],chunk:usize)->f32 {
    let rest=&row[chunk*NORM_CHUNK..];
    match rest.first_chunk::<NORM_CHUNK>() {Some(full)=>chunk_sum(full),None=>norm_square_sum(rest)}
}
/// `join` of 32 lane sums that lie in memory: 8 parts of 4 lanes, then `join_parts`. The same adds in the same
/// order as `join` (its distances 16 and 8 make the parts). A function of its own on purpose, as `lane_part`.
#[inline]
pub fn join_lanes(lanes:&[f32;LANES])->f32 {
    let mut parts=[0.0f32;APART];
    for (lane,part) in parts.iter_mut().enumerate() {*part=(lanes[lane]+lanes[lane+2*APART])+(lanes[lane+APART]+lanes[lane+3*APART]);}
    join_parts(parts)
}
/// The chunk sums of the norm tree, joined as `norm_squares` joins them: chunk `c` is added to lane `c mod 32`.
#[inline]
pub fn norm_join(parts:&[f32;LANES],chunks:usize)->f32 {
    let mut outer=[0.0f32;LANES];
    for (index,&part) in parts.iter().enumerate().take(chunks) {outer[index%LANES]+=part;}
    join_lanes(&outer)
}
/// Part `part` of the shared sum of row `row`. The caller has checked `shared_sum` and every extent with `lengths`.
#[inline(always)]
pub fn sum_part(s:Shape,row:usize,part:usize,x:[&[f32];4],w:&[f32])->f32 {
    let d=s.dim as usize;
    match s.op {
        Op::PlainNorm=>lane_squares(&x[0][row*d..(row+1)*d],part),
        Op::RoundedNorm=>norm_chunk(&x[0][row*d..(row+1)*d],part),
        _=>crate::block_attention::clip_lane(s,row,part,x,w),
    }
}
/// Write the words of row `row` from place `first` on, from the parts of the shared sum of the row: the words of `unit`.
#[inline(always)]
pub fn shared_words(s:Shape,row:usize,first:usize,x:[&[f32];4],w:&[f32],parts:&[f32;LANES],out:&mut[f32]) {
    match s.op {
        Op::PlainNorm=>crate::block_stream::normed(s,row,first,x,w,join_lanes(parts),out),
        Op::RoundedNorm=>crate::block_stream::normed(s,row,first,x,w,norm_join(parts,s.dim as usize/NORM_CHUNK),out),
        _=>crate::block_attention::clipped(s,row,first,x,w,join_lanes(parts),out),
    }
}
#[inline(always)]
fn attention(op:Op)->bool {matches!(op,Op::HeadClip|Op::RelBias|Op::WindowScores|Op::SinkSoftmax|Op::WindowValues)}
#[inline(always)]
fn stream(op:Op)->bool {matches!(op,Op::ShortConv|Op::MixBlend|Op::GateBlend|Op::PlainNorm|Op::RoundedNorm|Op::RowDots)}
#[inline(always)]
fn experts(op:Op)->bool {matches!(op,Op::CappedGate|Op::RouteFold|Op::Softcap|Op::RouterPick)}
#[inline(always)]
pub fn is_block(op:Op)->bool {attention(op) || stream(op) || experts(op)}
#[inline(always)]
pub fn lengths(s:Shape)->Option<Lengths> {
    if attention(s.op) {crate::block_attention::lengths(s)} else if stream(s.op) {crate::block_stream::lengths(s)} else if experts(s.op) {crate::block_experts::lengths(s)} else {None}
}
/// The counts of an arm with every check of `Shape::lengths` (finite numbers, no count above what the device ABI
/// holds): the counts the entries of the Inkling image (`kernels-inkling/src/lib.rs`, `run`) read their slices by. On the host
/// `Shape::lengths` gives the same counts (host test).
#[inline(always)]
pub fn checked_lengths(s:Shape)->Option<Lengths> {
    if !is_block(s.op) || !s.epsilon.is_finite() || !s.bound.is_finite() {return None;}
    (s.heads as usize).checked_mul(s.dim as usize)?;
    let (input,weight,units,words)=lengths(s)?;
    if input.into_iter().chain([weight,units.checked_mul(words)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
    Some((input,weight,units,words))
}
/// Write one output unit. The caller has checked every extent with `lengths`.
#[inline(always)]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    if attention(s.op) {crate::block_attention::unit(s,index,x,w,out)} else if stream(s.op) {crate::block_stream::unit(s,index,x,w,out)} else if experts(s.op) {crate::block_experts::unit(s,index,x,w,out)}
}
#[cfg(test)]
mod tests {

    /// The GPU list of the arms has the numbers and the layout of the host's: a recorded launch reads the same fields.
    #[test] fn the_gpu_list_of_the_arms_has_the_numbers_and_the_layout_of_the_host() {
        use device::Op as D;
        let pairs=[(D::HeadClip,Op::HeadClip),(D::RelBias,Op::RelBias),(D::ShortConv,Op::ShortConv),(D::WindowScores,Op::WindowScores),
            (D::SinkSoftmax,Op::SinkSoftmax),(D::WindowValues,Op::WindowValues),(D::MixBlend,Op::MixBlend),(D::GateBlend,Op::GateBlend),
            (D::CappedGate,Op::CappedGate),(D::RouteFold,Op::RouteFold),(D::Softcap,Op::Softcap),(D::PlainNorm,Op::PlainNorm),
            (D::RoundedNorm,Op::RoundedNorm),(D::RowDots,Op::RowDots),(D::RouterPick,Op::RouterPick)];
        for (gpu,host) in pairs {assert_eq!(gpu as u32,host as u32,"{host:?}");assert!(is_block(host));}
        // Every arm is in the list, and no older operation has an arm's number.
        assert_eq!((0..200u32).filter(|&n|n>=device::FIRST && n<device::FIRST+pairs.len() as u32).count(),pairs.len());
        assert_eq!(Op::HeadClip as u32,device::FIRST);
        assert_eq!((device::LAST-device::FIRST+1) as usize,pairs.len());
        assert_eq!(core::mem::offset_of!(device::RowsArguments,arguments),0);
        use core::mem::{offset_of,size_of};use crate::causal_ops::Arguments as H;
        assert_eq!((size_of::<device::Shape>(),size_of::<device::Arguments>()),(size_of::<Shape>(),size_of::<H>()));
        assert_eq!([offset_of!(device::Shape,heads),offset_of!(device::Shape,bound),offset_of!(device::Arguments,weight),offset_of!(device::Arguments,output),offset_of!(device::Arguments,shape)],
            [offset_of!(Shape,heads),offset_of!(Shape,bound),offset_of!(H,weight),offset_of!(H,output),offset_of!(H,shape)]);
    }
    /// The entry of the arms reads its slices by the counts of `Shape::lengths`.
    #[test] fn the_checked_counts_are_those_of_the_shape() {
        let s=|op,heads,dim,taps,rows,cols,epsilon,bound|Shape {op,heads,dim,taps,rows,cols,epsilon,bound};
        for shape in [s(Op::PlainNorm,3,64,0,0,0,1.0e-6,0.0),s(Op::RoundedNorm,2,2304,1,0,0,1.0e-6,0.0),s(Op::HeadClip,24,128,1,0,1,0.0,7.5),
            s(Op::RowDots,3,0,0,2,2304,0.0,0.0),s(Op::PlainNorm,3,64,0,0,0,f32::NAN,0.0),s(Op::PlainNorm,0,64,0,0,0,1.0e-6,0.0),s(Op::Project,1,4,0,0,4,0.0,0.0)] {
            assert_eq!(checked_lengths(shape),if is_block(shape.op) {shape.lengths()} else {None},"{shape:?}");
        }
    }
    use super::*;
    const ARMS:[Op;15]=[Op::HeadClip,Op::RelBias,Op::ShortConv,Op::WindowScores,Op::SinkSoftmax,Op::WindowValues,Op::MixBlend,Op::GateBlend,Op::CappedGate,Op::RouteFold,Op::Softcap,Op::PlainNorm,Op::RoundedNorm,Op::RowDots,Op::RouterPick];
    #[test] fn the_arms_are_block_arms_and_each_has_one_file() {
        for op in ARMS {
            assert!(is_block(op));
            assert_eq!(u8::from(attention(op))+u8::from(stream(op))+u8::from(experts(op)),1);
        }
        for op in [Op::Conv,Op::Softmax,Op::Route,Op::PackBf16,Op::AbsorbProject] {assert!(!is_block(op));}
    }
    #[test] fn a_shape_with_no_size_is_refused_for_every_arm() {
        for op in ARMS {assert!(Shape {op,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}.lengths().is_none());}
    }
    #[test] fn the_arm_names_come_after_every_older_name() {
        assert_eq!(Op::HeadClip as u32,Op::AbsorbProject as u32+1);
        assert_eq!(Op::RoundedNorm as u32,Op::AbsorbProject as u32+13);
        assert_eq!(Op::RowDots as u32,Op::AbsorbProject as u32+14);
        assert_eq!(Op::RouterPick as u32,Op::AbsorbProject as u32+15);
    }
    /// The norm tree as its definition says it, with the chunk sums kept in a list.
    fn norm_tree_by_definition(x:&[f32])->f32 {
        fn sum(mut lanes:[f32;32])->f32 {for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32 {lanes[lane]=old[lane]+old[lane^distance];}}lanes[0]}
        let partials=x.chunks(128).map(|chunk|{let mut lanes=[0.0;32];for i in 0..chunk.len() {lanes[i%32]+=chunk[i]*chunk[i];}sum(lanes)}).collect::<Vec<_>>();
        let mut lanes=[0.0;32];for i in 0..partials.len() {lanes[i%32]+=partials[i];}
        sum(lanes)
    }
    #[test] fn the_norm_tree_keeps_its_order_at_every_width() {
        for width in [1usize,2,31,32,33,127,128,129,2304,4096,4097,5000] {
            // Values of many sizes, so another order of the sums gives another last bit.
            let x=(0..width).map(|i|((i*37%101) as f32-50.0)*[1.0e-3,1.0,3.0e3,7.0e-2][i%4]).collect::<Vec<_>>();
            assert_eq!(norm_square_sum(&x).to_bits(),norm_tree_by_definition(&x).to_bits(),"width {width}");
        }
        // Hand-worked, one chunk. Lane 0 holds 4096^2 = 16777216 and then 1^2: the 1 is lost in FP32. Lane 1 holds
        // one 1. The join adds lane 1 to lane 0 at the end: that 1 is lost too.
        let mut x=vec![0.0f32;64];x[0]=4096.0;x[32]=1.0;x[1]=1.0;
        assert_eq!(norm_square_sum(&x),16_777_216.0);
        // Lanes 1 and 3 hold one 1 each. The join adds lane 3 to lane 1 first (distance 2), then that 2 to lane 0
        // (distance 1): 16777216 + 2 is exact. A chain in rising index would lose both.
        let mut x=vec![0.0f32;64];x[0]=4096.0;x[1]=1.0;x[3]=1.0;
        assert_eq!(norm_square_sum(&x),16_777_218.0);
    }
    /// The GPU schedule of a row that shares its sum, played on the host: in each block the parts, then a word a thread.
    #[test] fn a_shared_row_sum_gives_the_words_of_the_unit() {
        let mut seed=0x9e37_79b9u32;
        let mut next=||{seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;((seed>>8) as f32/8388608.0-1.0)*if seed&64==0 {1.0} else {1.0e3}};
        let mut seen=0;
        for op in [Op::PlainNorm,Op::RoundedNorm,Op::HeadClip] {for d in [128usize,256,384,1024,2304,3968,4096] {for h in [1usize,2,3,WHOLE_ROWS-1,WHOLE_ROWS,WHOLE_ROWS+1] {for form in 0..4u32 {
            if h>=WHOLE_ROWS-1 && !matches!(d,128|2304) {continue;}
            let s=match op {
                Op::HeadClip=>Shape {op,heads:h as u32,dim:d as u32,taps:form&1,rows:0,cols:form>>1,epsilon:0.0,bound:if form==3 {1.0e-3} else {7.5}},
                _ if form<2=>Shape {op,heads:h as u32,dim:d as u32,taps:if op==Op::RoundedNorm {form} else {0},rows:0,cols:0,epsilon:1.0e-6,bound:0.0},
                _=>continue,
            };
            let (lengths,weight,units,words)=s.lengths().unwrap();
            let inputs:[Vec<f32>;4]=core::array::from_fn(|i|(0..lengths[i]).map(|_|next()).collect());
            let x:[&[f32];4]=core::array::from_fn(|i|&inputs[i][..]);
            let w:Vec<f32>=(0..weight).map(|_|next()).collect();
            let (blocks,parts)=shared_sum(s).unwrap();let span=shared_span(s);
            assert_eq!(span,if h>=WHOLE_ROWS {d} else {SUM_THREADS});
            assert!(span.is_multiple_of(SUM_THREADS) && d.is_multiple_of(span));
            assert_eq!((blocks*span,parts<=LANES),(units*words,true));
            assert_eq!(geometry(s,units),(blocks as u32,SUM_THREADS as u32));
            let mut alone=vec![f32::NAN;units*words];
            for (index,slice) in alone.chunks_exact_mut(words).enumerate() {crate::causal_ops::unit(s,index,x,&w,slice);}
            // Each output word is written once: by thread t of block b at b * span + t * each + k, k < each = span / SUM_THREADS.
            let mut written=vec![0u8;units*words];
            for block in 0..blocks {
                let (row,first)=split(block*span,d);
                let mut sums=[0.0f32;LANES];
                for (part,sum) in sums.iter_mut().enumerate().take(parts) {*sum=sum_part(s,row,part,x,&w);}
                let each=span/SUM_THREADS;
                for thread in 0..SUM_THREADS {
                    let at=block*span+thread*each;
                    let mut words=vec![f32::NAN;each];shared_words(s,row,first+thread*each,x,&w,&sums,&mut words);
                    for (k,word) in words.iter().enumerate() {
                        assert_eq!(word.to_bits(),alone[at+k].to_bits(),"{s:?} block {block} thread {thread} k {k}");seen+=1;written[at+k]+=1;
                    }
                }
            }
            assert!(written.iter().all(|&n|n==1),"{s:?}: a word written other than once");
        }}}}
        assert!(seen>100_000);
        // The join of lanes in memory is `join`.
        for _ in 0..1000 {let lanes:[f32;LANES]=core::array::from_fn(|_|next());assert_eq!(join_lanes(&lanes).to_bits(),join(lanes).to_bits());}
        // No shared sum: a width that is no multiple of a block, a norm tree with more chunks than lanes, another arm.
        let s=|op,d|Shape {op,heads:1,dim:d,taps:0,rows:0,cols:0,epsilon:1.0e-6,bound:1.0};
        assert!(shared_sum(s(Op::PlainNorm,96)).is_none() && shared_sum(s(Op::RoundedNorm,4224)).is_none() && shared_sum(s(Op::PlainNorm,4224)).is_some() && shared_sum(s(Op::GateBlend,128)).is_none());
    }
    #[test] fn the_register_form_of_the_norm_tree_gives_the_bits_of_its_definition() {
        assert_eq!((norm_part(2304),norm_part(128),norm_part(32),norm_part(64),norm_part(5),norm_part(33),norm_part(1)),(32,32,32,32,5,33,1));
        // Values of many sizes and both signs, so another order of the sums gives another last bit.
        let mut state=0x2545_f491u32;
        let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
        for width in [1usize,2,31,32,33,64,96,127,128,129,160,255,256,257,1000,1024,2304,4095,4096,4097,4128,5000,8192,8200] {
            for round in 0..24 {
                let x=(0..width).map(|_|{let bits=next();let value=((bits>>8) as f32/8_388_608.0-1.0)*[1.0e-3,1.0,3.0e3,7.0e-2][(bits&3) as usize];
                    if round%3==0 && bits%7==0 {0.0} else {value}}).collect::<Vec<_>>();
                assert_eq!(norm_squares(&x).to_bits(),norm_square_sum(&x).to_bits(),"width {width} round {round}");
                assert_eq!(norm_squares(&x).to_bits(),norm_tree_by_definition(&x).to_bits(),"width {width} round {round}");
                assert_eq!(row_squares(&x).to_bits(),crate::causal_ops::square_sum(&x).to_bits(),"32-lane tree, width {width} round {round}");
            }
        }
        // The hand-worked rows of the test above: the 1 that is lost in lane 0, and the two that the join keeps.
        let mut x=vec![0.0f32;128];x[0]=4096.0;x[32]=1.0;x[1]=1.0;
        assert_eq!(norm_squares(&x),16_777_216.0);
        let mut x=vec![0.0f32;128];x[0]=4096.0;x[1]=1.0;x[3]=1.0;
        assert_eq!(norm_squares(&x),16_777_218.0);
        // The join in parts against the join itself, on lane sums that round at every step.
        for _ in 0..200 {
            let lanes:[f32;LANES]=core::array::from_fn(|_|{let bits=next();((bits>>8) as f32/8_388_608.0-1.0)*[1.0e-3,1.0,3.0e3,7.0e7][(bits&3) as usize]});
            let parts:[f32;APART]=core::array::from_fn(|l|(lanes[l]+lanes[l+16])+(lanes[l+8]+lanes[l+24]));
            assert_eq!(join_parts(parts).to_bits(),join(lanes).to_bits());
        }
        // `lane_sum` from a place that is not zero, with a last row that is not full.
        struct Listed<'a>(&'a [f32]);
        impl Terms for Listed<'_> {fn term(&self,at:usize)->f32 {self.0[at]}}
        for (first,end) in [(0usize,0usize),(0,1),(0,32),(0,33),(64,64),(64,65),(64,200),(1024,2048),(1024,2047),(32,1000)] {
            let terms=(0..end).map(|_|{let bits=next();((bits>>8) as f32/8_388_608.0-1.0)*[1.0e-3,1.0,3.0e3,7.0e7][(bits&3) as usize]}).collect::<Vec<_>>();
            let mut lanes=[0.0f32;LANES];
            for at in first..end {lanes[(at-first)%LANES]+=terms[at];}
            assert_eq!(lane_sum(&Listed(&terms),first,end).to_bits(),join(lanes).to_bits(),"places {first} to {end}");
        }
        // The 32-bit split of a unit index.
        for (index,size) in [(0usize,1usize),(5,1),(0,7),(6,7),(7,7),(8,7),(110_591,2304),(24_575,1024),(u32::MAX as usize-1,3),(u32::MAX as usize-1,u32::MAX as usize)] {
            assert_eq!(split(index,size),(index/size,index%size),"{index} by {size}");
        }
        // One full chunk alone.
        let chunk:[f32;NORM_CHUNK]=core::array::from_fn(|i|((i*37%101) as f32-50.0)*[1.0e-3,1.0,3.0e3,7.0e-2][i%4]);
        assert_eq!(norm_squares(&chunk).to_bits(),norm_tree_by_definition(&chunk).to_bits());
    }
}
