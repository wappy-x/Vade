//! Block arms of attention with a window, a relative position term and a sink: `HeadClip`, `RelBias`,
//! `WindowScores`, `SinkSoftmax`, `WindowValues`. Safe shared code for the host and the GPU.
//!
//! Shape fields: `h` = heads, `d` = dim, `t` = taps, `r` = rows, `k` = cols. Counts are 32-bit words.
//! "Packed" = two BF16 values a word, as `PackBf16` writes them. A field that an arm does not use must be zero.
//! - `HeadClip`: `h` heads of `d` values; `t` = 1 if a scale is given; `k` = 1 if a factor is given; `bound` = the
//!   clip threshold. Inputs: `x` [h*d], `factor` [k]. Weight: `scale` [t*h*d]. Units `h * d / p`, words `p`, where
//!   `p` = `block_ops::norm_part(d)`: 32 words when the width is a multiple of 32, else the whole head. A unit is
//!   one part of one head, and each unit makes the length of the whole head itself, in the one fixed order.
//! - `RelBias`: `h` heads, `d` = width of the relative state, `k` = extent of the table. Input: the state [h*d].
//!   Weight: the table [d*k], row then distance. Units `h * k / p`, words `p`, where `p` = `rel_part(k)`: 8
//!   distances when the extent is a multiple of 8, else 1. Each word is the same chain over the state in any layout.
//! - `WindowScores`: `h` query heads of `d` values, `t` = query heads for each key head, `r` rows of the key plane,
//!   `k` = extent of the bias; `bound` = the score scale. Inputs: the query [h*d], the key plane [r*(h/t)*d/2] packed,
//!   the bias [h*k], the step words [2] = live count and slot of the current token. Units `h*r`, words 1.
//! - `SinkSoftmax`: `h` heads, `r` rows. Inputs: the scores [h*r], the sink [h], the step word [1] = live count.
//!   Units `h * r / p`, words `p`, where `p` = `softmax_part(r)`: 32 words when the rows are a multiple of 32, else
//!   the whole head. A unit is one part of one head, so many GPU threads write a head. Each of them finds the
//!   largest score and makes the sum of the head itself, in the one fixed order: every part divides by the same bits.
//! - `WindowValues`: as `WindowScores` without `k` and `bound`. Inputs: the weights [h*r], the value plane
//!   [r*(h/t)*d/2] packed, the gate [h*d], the step words [2]. Units `h*d`, words 1.
//!
//! The row of distance `dist` from the current token is `(slot - dist) mod r`, in a ring and in a straight plane.
//! The order of a sum over keys depends on the distance only, never on the slot: see `distance_sum`.
//! A step word is a count held as a float. A word that is no whole number, a live count above the rows or a slot
//! outside the rows gives NaN words. The host form refuses a NaN result, so such a launch is never used.
use crate::{block_ops::{lane_sum,norm_part,split,Lengths,Op,Shape,Terms,root},fixed_math::{exp,rn,sigmoid}};

/// The clip adds this to the length of a head before the division.
pub const CLIP_EPSILON:f32=1e-6;
/// Distances in one block of a sum over keys. A part of the sum order, not a size of a model.
pub const BLOCK:usize=1024;
/// Words of one unit of `SinkSoftmax` when the rows are a multiple of it. A part of the launch layout.
pub const SOFTMAX_PART:usize=32;
/// The words of one unit of `SinkSoftmax` with `r` rows: `SOFTMAX_PART`, or the whole head when `r` is no multiple.
#[inline(always)]
pub fn softmax_part(r:usize)->usize {if r.is_multiple_of(SOFTMAX_PART) {SOFTMAX_PART} else {r}}
/// Distances of one unit of `RelBias` when the extent is a multiple of it (read arm 4). A part of the launch layout:
/// each word is the same chain as before, so the words do not depend on it.
pub const REL_PART:usize=8;
/// The distances of one unit of `RelBias` with extent `k`: `REL_PART`, or 1 when `k` is no multiple.
#[inline(always)]
pub fn rel_part(k:usize)->usize {if k.is_multiple_of(REL_PART) {REL_PART} else {1}}
/// The largest row count whose live count and slot are exact as floats.
const ROWS_MAX:usize=1<<24;

/// Word counts of an arm of this file. `None` refuses the shape.
#[inline]
pub fn lengths(s:Shape)->Option<Lengths> {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    let width=h.checked_mul(d)?;
    let plane=|| if d.is_multiple_of(2) && t>0 && h.is_multiple_of(t) && r>0 && r<=ROWS_MAX {r.checked_mul(h/t)?.checked_mul(d/2)} else {None};
    match s.op {
        Op::HeadClip if width>0 && t<=1 && k<=1 && r==0 && s.bound>0.0 && s.epsilon==0.0=>Some(([width,k,0,0],t*width,h.checked_mul(d/norm_part(d))?,norm_part(d))),
        Op::RelBias if width>0 && k>0 && t==0 && r==0 && s.bound==0.0 && s.epsilon==0.0=>Some(([width,0,0,0],d.checked_mul(k)?,h.checked_mul(k/rel_part(k))?,rel_part(k))),
        Op::WindowScores if width>0 && k>0 && s.bound>0.0 && s.epsilon==0.0=>Some(([width,plane()?,h.checked_mul(k)?,2],0,h.checked_mul(r)?,1)),
        Op::SinkSoftmax if h>0 && r>0 && r<=ROWS_MAX && d==0 && t==0 && k==0 && s.bound==0.0 && s.epsilon==0.0=>Some(([h.checked_mul(r)?,h,1,0],0,h.checked_mul(r/softmax_part(r))?,softmax_part(r))),
        Op::WindowValues if width>0 && k==0 && s.bound==0.0 && s.epsilon==0.0=>Some(([h.checked_mul(r)?,plane()?,width,2],0,width,1)),
        _=>None,
    }
}
/// The factor of the clip: 1 for a head at or under the threshold, else the threshold over the length.
/// A number divided by a 32-bit tensor is two rounded steps in the reference: the 32-bit reciprocal of each value,
/// then the product with the number as 32-bit (measured on the reference, receipt `divform2.json` of its contract A).
#[inline(always)]
fn clip_factor(length:f32,threshold:f32)->f32 {
    let reciprocal=1.0/(length+CLIP_EPSILON);
    let factor=threshold*reciprocal;
    if factor>1.0 {1.0} else {factor}
}
/// A step word as a count, if it is a whole number that is not above `most`.
#[inline(always)]
fn count(word:f32,most:usize)->Option<usize> {let n=word as usize;if n as f32==word && n<=most {Some(n)} else {None}}
/// `live` for the staged rows form of `WindowScores` (`block_scores`).
#[inline(always)]
pub(crate) fn live_of(step:&[f32],rows:usize)->Option<(usize,usize)> {live(step,rows)}
/// The live count and the slot of the current token, if the count is not above the rows and the slot is a row.
#[inline(always)]
fn live(step:&[f32],rows:usize)->Option<(usize,usize)> {
    match (count(step[0],rows),count(step[1],rows-1)) {(Some(n),Some(slot))=>Some((n,slot)),_=>None}
}
/// The sum of `term(dist)` for every distance below `n`, in the one fixed order of every sum over keys.
/// The distances are cut into blocks of `BLOCK`. In a block, the term of distance `dist` goes to lane
/// `dist mod LANES`, and each lane is a chain in rising distance from +0.0. The lanes of a block are joined.
/// The block sums are a chain in rising block number from +0.0.
/// So the order is a function of the distance alone. It does not know where a ring starts.
/// This is the definition. The arms use `distance_tree`, which gives the same bits (tested).
#[cfg(test)]
pub(crate) fn distance_sum(n:usize,term:impl Fn(usize)->f32)->f32 {
    let (mut total,mut first)=(0.0f32,0);
    while first<n {
        let end=if n-first>BLOCK {first+BLOCK} else {n};
        let mut lanes=[0.0f32;crate::block_ops::LANES];
        let mut dist=first;
        while dist<end {lanes[dist%crate::block_ops::LANES]+=term(dist);dist+=1;}
        total+=crate::block_ops::join(lanes);
        first=end;
    }
    total
}
/// `distance_sum` with every partial sum in a register: the same terms, the same sums, the same order.
/// A block starts at a multiple of `LANES`, so the lane of `lane_sum` is `dist mod LANES`.
#[inline(always)]
pub(crate) fn distance_tree<T:Terms>(n:usize,terms:&T)->f32 {
    let (mut total,mut first)=(0.0f32,0);
    while first<n {
        let end=if n-first>BLOCK {first+BLOCK} else {n};
        total+=lane_sum(terms,first,end);
        first=end;
    }
    total
}
/// One head of the clip: value `j` is the input read as BF16, times its scale when a scale is given.
struct Scaled<'a> {x:&'a [f32],scale:Option<&'a [f32]>}
impl Scaled<'_> {
    #[inline(always)]
    fn value(&self,j:usize)->f32 {let value=rn(self.x[j]);match self.scale {Some(scale)=>value*scale[j],None=>value}}
}
/// The squares of the values of one head, for the 32-lane tree.
impl Terms for Scaled<'_> {
    #[inline(always)]
    fn term(&self,j:usize)->f32 {let value=self.value(j);value*value}
}
/// The terms of the softmax sum of one head: `exp(score - top)`.
struct Exps<'a> {scores:&'a [f32],top:f32}
impl Terms for Exps<'_> {
    #[inline(always)]
    fn term(&self,dist:usize)->f32 {exp(self.scores[dist]-self.top)}
}
/// The terms of the value sum of one query head and one channel: the weight of the distance times the value of
/// its row. `stride` = packed words of one row of the plane, `place` = the packed word of the channel in a row,
/// `low` = the channel is the low half of its word.
struct Weighted<'a> {weights:&'a [f32],plane:&'a [f32],rows:usize,slot:usize,stride:usize,place:usize,low:bool}
impl Terms for Weighted<'_> {
    #[inline(always)]
    fn term(&self,dist:usize)->f32 {
        // The row of distance `dist` is `(slot - dist) mod rows`. Both are below `rows`, so one test gives it.
        let row=if dist<=self.slot {self.slot-dist} else {self.slot+self.rows-dist};
        let bits=self.plane[row*self.stride+self.place].to_bits();
        self.weights[dist]*f32::from_bits(if self.low {bits<<16} else {bits&0xffff_0000})
    }
}
/// Words of head `head` of `HeadClip` from place `first` on, from the sum of the squares of the head.
#[inline(always)]
pub fn clipped(s:Shape,head:usize,first:usize,x:[&[f32];4],w:&[f32],squares:f32,out:&mut[f32]) {
    let (d,t,k)=(s.dim as usize,s.taps as usize,s.cols as usize);let base=head*d;
    let values=Scaled {x:&x[0][base..base+d],scale:if t==1 {Some(&w[base..base+d])} else {None}};
    let factor=clip_factor(root(squares),s.bound);
    for (offset,word) in out.iter_mut().enumerate() {
        let clipped=rn(values.value(first+offset)*factor);
        *word=if k==1 {rn(clipped*x[1][0])} else {clipped};
    }
}
/// The chain of lane `lane` of the squares of head `head`: rising place from +0.0 (a lane of `lane_sum`).
#[inline]
pub fn clip_lane(s:Shape,head:usize,lane:usize,x:[&[f32];4],w:&[f32])->f32 {
    let (d,base)=(s.dim as usize,head*s.dim as usize);
    let values=Scaled {x:&x[0][base..base+d],scale:if s.taps==1 {Some(&w[base..base+d])} else {None}};
    let mut sum=0.0f32;let mut j=lane;while j<d {sum+=values.term(j);j+=crate::block_ops::LANES;}sum
}
// The staged forms of the GPU entry (`kernels-inkling`): the threads of one block share the sums of one head, so a
// head is no longer one long chain in each of its threads. The terms, the chains and the joins are those of `unit`,
// so the words are (host tests in `block_attention_tests.rs`).
/// `SinkSoftmax` staged (`Some(blocks)`, one a head): a block of `block_ops::SUM_THREADS` threads a head, any row
/// count. Thread `t` finds the best score of the distances `t, t + threads, ..` (`softmax_best`); thread 0 picks the
/// top from them and the sink (`softmax_pick`), which is the top of `unit`; then `SOFTMAX_GROUP` blocks of distances
/// at a time, each with `block_ops::APART` threads, one for each part of its sum (`softmax_part_sum`); every thread
/// chains the block sums (`softmax_add`) and writes the weights of its distances (`softmax_word`).
#[inline]
pub fn softmax_stage(s:Shape)->Option<usize> {
    if s.op!=Op::SinkSoftmax {return None;}
    crate::block_ops::checked_lengths(s)?;
    Some(s.heads as usize)
}
/// `SinkSoftmax` at rows staged (`Some(blocks)`, the entry `resident_launch_block_softmax`): block `b` is head
/// `b mod h` of row `b / h`, with `block_ops::SUM_THREADS` threads, and runs the staged form of one head on the inputs
/// of its row (`block_rows::row_inputs`), so its words are those of the rows entry. The scores must be each row's own
/// (bit 0 of the mask clear); the sink and the step word may be shared or per row; no list or window form.
#[inline]
pub fn softmax_rows_stage(s:Shape,rows:usize,shared:u32)->Option<usize> {
    if s.op!=Op::SinkSoftmax || shared&!0b0110!=0 {return None;}
    crate::block_rows::lengths(s,rows,shared)?;
    let blocks=rows.checked_mul(s.heads as usize)?;
    (blocks<=u32::MAX as usize).then_some(blocks)
}
/// Blocks of distances whose parts a staged `SinkSoftmax` block makes at one time.
pub const SOFTMAX_GROUP:usize=crate::block_ops::SUM_THREADS/crate::block_ops::APART;
/// The live count of a `SinkSoftmax` launch, as `unit` reads it; `None` makes every word NaN.
#[inline(always)]
pub fn softmax_live(s:Shape,x:[&[f32];4])->Option<usize> {count(x[2][0],s.rows as usize)}
/// The sink of head `head`, as `unit` reads it.
#[inline(always)]
pub fn softmax_sink(head:usize,x:[&[f32];4])->f32 {rn(x[1][head])}
/// No candidate: a best that every score above minus infinity beats and that loses every tie.
pub const NO_BEST:(f32,u32)=(f32::NEG_INFINITY,u32::MAX);
/// The best of the live scores `v(dist)` at `dist = start, start + step, ..` below `n`: the first of them (in rising
/// distance) whose value is the largest, as the chain of `unit` keeps it (a NaN never wins); `NO_BEST` if none.
#[inline(always)]
pub fn softmax_best(n:usize,start:usize,step:usize,v:&impl Fn(usize)->f32)->(f32,u32) {
    let mut best=NO_BEST;let mut dist=start;
    while dist<n {let score=v(dist);if score>best.0 {best=(score,dist as u32);}dist+=step;}
    best
}
/// The top of `unit` from the sink and the bests of all the threads: `unit` starts from the sink and takes a score
/// only when it is larger, so its top is the sink when the sink is NaN, else the first value (the sink before every
/// score, then in rising distance) equal to the largest. A best equal to the top so far wins only from a lower
/// distance; the sink is before every distance.
#[inline(always)]
pub fn softmax_pick(sink:f32,bests:impl Iterator<Item=(f32,u32)>)->f32 {
    if sink.is_nan() {return sink;}
    let mut top=(sink,None::<u32>);
    for (value,dist) in bests {
        let earlier=match top.1 {None=>false,Some(at)=>dist<at};
        if value>top.0 || value==top.0 && earlier {top=(value,Some(dist));}
    }
    top.0
}
/// The term of distance `dist` of the softmax sum, as `Exps` makes it.
#[inline(always)]
pub fn softmax_exp(score:f32,top:f32)->f32 {exp(score-top)}
/// Terms read through a function.
struct By<F>(F);
impl<F:Fn(usize)->f32> Terms for By<F> {
    #[inline(always)]
    fn term(&self,at:usize)->f32 {(self.0)(at)}
}
/// Part `part` (below `block_ops::APART`) of the sum of the exps `e(dist)` of the distances `first..end` (a block of
/// `distance_tree`): the lanes of `lane_sum`.
#[inline(always)]
pub fn softmax_part_sum(first:usize,end:usize,part:usize,e:&impl Fn(usize)->f32)->f32 {crate::block_ops::lane_part(&By(e),first,end,part)}
/// The chain of `distance_tree`: the next block sum, joined from its parts as `lane_sum` joins them.
#[inline(always)]
pub fn softmax_add(total:f32,parts:[f32;crate::block_ops::APART])->f32 {total+crate::block_ops::join_parts(parts)}
/// The total of `unit`: the chained block sums (from +0.0), then the sink's term.
#[inline(always)]
pub fn softmax_total(chain:f32,sink:f32,top:f32)->f32 {chain+exp(sink-top)}
/// The weight of distance `dist` from its exp `e` (`softmax_exp`), as `unit` writes it.
#[inline(always)]
pub fn softmax_word(dist:usize,n:usize,e:f32,total:f32)->f32 {if dist<n {rn(e/total)} else {0.0}}
/// Output words of one block of a staged `WindowValues`: `VALUES_CHANNELS` channels, each with `block_ops::APART`
/// threads, one for each part of its sum.
pub const VALUES_CHANNELS:usize=crate::block_ops::SUM_THREADS/crate::block_ops::APART;
/// `WindowValues` staged (`Some(blocks)`) when the dim is a multiple of `VALUES_CHANNELS`: block `b` writes the words
/// `b * VALUES_CHANNELS ..` of one head. Thread `t` makes part `t / VALUES_CHANNELS` of the sum of channel
/// `t mod VALUES_CHANNELS` in each block of distances (`values_part`); the block meets, and the thread of part 0 joins
/// the parts in the order of `lane_sum`, adds the block sums as `distance_tree` does, and writes the word
/// (`values_word`).
#[inline]
pub fn values_stage(s:Shape)->Option<usize> {
    if s.op!=Op::WindowValues || s.dim==0 || !(s.dim as usize).is_multiple_of(VALUES_CHANNELS) {return None;}
    let (_,_,units,_)=crate::block_ops::checked_lengths(s)?;
    Some(units/VALUES_CHANNELS)
}
/// The live count and slot of a `WindowValues` launch, as `unit` reads them; `None` makes every word NaN.
#[inline(always)]
pub fn values_live(s:Shape,x:[&[f32];4])->Option<(usize,usize)> {live(x[3],s.rows as usize)}
/// Part `part` of the sum of output word `index` over the distances `first..end` (a block of `distance_tree`).
#[inline(always)]
pub fn values_part(s:Shape,index:usize,slot:usize,x:[&[f32];4],first:usize,end:usize,part:usize)->f32 {
    let (h,d,t,r)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize);
    let (head,channel)=split(index,d);let half=d/2;
    let terms=Weighted {weights:&x[0][head*r..(head+1)*r],plane:x[1],rows:r,slot,stride:(h/t)*half,place:(head/t)*half+channel/2,low:channel.is_multiple_of(2)};
    crate::block_ops::lane_part(&terms,first,end,part)
}
/// Output word `index` of `WindowValues` from the chain of its block sums, as `unit` writes it.
#[inline(always)]
pub fn values_word(s:Shape,index:usize,x:[&[f32];4],total:f32)->f32 {
    let d=s.dim as usize;let (head,channel)=split(index,d);
    rn(rn(total)*sigmoid(rn(x[2][head*d+channel])))
}
/// Write one output unit of an arm of this file. The caller has checked every extent with `lengths`.
#[inline]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    match s.op {
        // One head. The input is read as BF16. The scale product, the length and the factor are 32-bit.
        // The result is rounded after the clip, and once more after the factor of the position.
        // The squares of the whole head on the 32-lane tree of `causal_ops::square_sum`: lane = index mod 32.
        Op::HeadClip=>{
            let part=norm_part(d);let (head,first)=split(index,d/part);let base=head*d;
            let values=Scaled {x:&x[0][base..base+d],scale:if t==1 {Some(&w[base..base+d])} else {None}};
            clipped(s,head,first*part,x,w,lane_sum(&values,0,d),out)
        },
        // One head and `rel_part(k)` neighbour distances; each word a chain over the width of the state, rounded once.
        Op::RelBias=>{
            let part=rel_part(k);let (head,first)=split(index,k/part);let (first,state)=(first*part,&x[0][head*d..(head+1)*d]);
            for (i,word) in out.iter_mut().enumerate().take(part) {
                let mut sum=0.0f32;
                for (j,&v) in state.iter().enumerate() {sum+=rn(v)*w[j*k+first+i];}
                *word=rn(sum);
            }
        },
        // One query head and one distance. Rounded after the dot, after the scale and after the bias.
        // The bias is +0.0 at a distance outside its extent. A slot that is not live gets 0.0 and reads no key.
        Op::WindowScores=>{
            let (head,dist)=split(index,r);
            out[0]=match live(x[3],r) {
                None=>f32::NAN,
                Some((n,_)) if dist>=n=>0.0,
                Some((_,slot))=>{
                    // The row of distance `dist` is `(slot - dist) mod r`. Both are below `r`, so one test gives it.
                    let half=d/2;let row=if dist<=slot {slot-dist} else {slot+r-dist};
                    let start=(row*(h/t)+head/t)*half;
                    let key=&x[1][start..start+half];let query=&x[0][head*d..(head+1)*d];
                    // A packed word holds two values of the key: value `2i` in the low half, value `2i+1` in the high half.
                    let mut sum=0.0f32;
                    for (pair,word) in query.as_chunks::<2>().0.iter().zip(key) {
                        let bits=word.to_bits();
                        sum+=rn(pair[0])*f32::from_bits(bits<<16);
                        sum+=rn(pair[1])*f32::from_bits(bits&0xffff_0000);
                    }
                    let scaled=rn(rn(sum)*s.bound);
                    rn(scaled+if dist<k {x[2][head*k+dist]} else {0.0})
                },
            };
        },
        // One query head: the largest of the live scores and the sink is taken out, then `exp`, then the sum in the
        // fixed order with the sink added last, then one division for each weight. Each weight is rounded.
        Op::SinkSoftmax=>{
            let Some(n)=count(x[2][0],r) else {out.fill(f32::NAN);return;};
            let part=softmax_part(r);let (head,first)=split(index,r/part);let first=first*part;
            let scores=&x[0][head*r..(head+1)*r];
            let sink=rn(x[1][head]);
            let mut top=sink;
            for &score in &scores[..n] {if score>top {top=score;}}
            let total=distance_tree(n,&Exps {scores:&scores[..n],top})+exp(sink-top);
            for (offset,weight) in out.iter_mut().enumerate() {let dist=first+offset;*weight=if dist<n {rn(exp(scores[dist]-top)/total)} else {0.0};}
        },
        // One query head and one channel: the sum over the live keys in the fixed order, rounded, then the gate.
        Op::WindowValues=>{
            let (head,channel)=split(index,d);
            out[0]=match live(x[3],r) {
                None=>f32::NAN,
                Some((n,slot))=>{
                    let half=d/2;
                    let terms=Weighted {weights:&x[0][head*r..(head+1)*r],plane:x[1],rows:r,slot,stride:(h/t)*half,place:(head/t)*half+channel/2,low:channel.is_multiple_of(2)};
                    rn(rn(distance_tree(n,&terms))*sigmoid(rn(x[2][head*d+channel])))
                },
            };
        },
        _=>{},
    }
}
