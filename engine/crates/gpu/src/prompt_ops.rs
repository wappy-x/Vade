//! Layer-major prompt rows for one chat. One launch serves many prompt rows.
//! Each output element keeps the scalar operation's fold and rounding order.
//! Host and device share this code. Sequential state stays in token order.
use crate::causal_ops::{self,Op as Causal,Shape as CausalShape};
use crate::next_math::{exp,bf16};
#[inline(always)]
pub(crate) fn round(x:f32)->f32 {f32::from_bits(bf16(x)<<16)}
#[inline(always)]
fn sigmoid(x:f32)->f32 {1.0/(1.0+exp(-x))}
#[inline(always)]
pub(crate) fn root(x:f32)->f32 {
    #[cfg(target_arch="nvptx64")] {crate::qsa_common::root(x)}
    #[cfg(not(target_arch="nvptx64"))] {x.sqrt()}
}
/// Largest prompt group one launch accepts. The checks below keep all address
/// products inside the existing 32-bit element ABI.
pub const MAX_ROWS:usize=1<<16;
#[repr(u32)]
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Op {
    /// One fixed-shape causal operation for each token. `inner` gives the shape.
    Wrapped,
    /// Plain weight rows `[rows,cols]` over token rows `[tokens,cols]`. Output `[rows,tokens]`.
    ProjectRows,
    /// Output `[cols,rows]` from input `[rows,cols]`. Bits move and nothing rounds.
    Transpose,
    /// Output `[tokens,dim]` is input `[tokens,dim]` times the shared weight `[dim]`.
    /// `cols` nonzero adds the BF16 boundary of a BF16 norm parameter.
    NormApply,
    /// Output `[tokens,3,heads*dim]`. Inputs are Q, K, V `[tokens,heads*dim]` and history.
    ConvRows,
    /// Output `[3,taps-1,heads*dim]`: the history after the last token.
    HistoryRows,
    /// Output `[tokens,heads,dim]`: each key's decay. It does not depend on the state.
    StepDecay,
    /// One thread for each value column, carrying every token in order. Output
    /// `[heads,dim,dim]` then `[tokens,heads,dim]`: the state after the last token, then
    /// each token's recurrent output. Lanes of one warp share cache lines there.
    StepRows,
    /// Output `[tokens,pools]`. Each token scores only the pools complete at its context.
    PoolScoresRows,
    /// Output `[tokens,slots+1]`. Each token selects from its own context.
    SelectRows,
    /// Output `[tokens,heads,cols]`. Keys come from the shared expanded latent rows.
    ScoresRows,
    /// Output `[tokens,heads,cols]`.
    SoftmaxRows,
    /// Output `[tokens,heads,taps]`. Values come from the shared expanded latent rows.
    ValuesRows,
    /// Output `[rows,cols]` where output row `i` copies input row `table[i]`.
    /// `first` gives the number of input rows.
    PermuteRows,
    /// Output `[rows,cols]` from experts' gathered units. `slots` words per rank,
    /// `taps` ranks, `dim` local weight rows. Table `[rows,3]`: earlier token rows,
    /// this expert's tokens, this row's token index in the expert.
    ExpertPlace,
    /// Output as `ScoresRows`. Fourth input `[rows,tokens]`: the token's slot for that row, or -1.
    /// Each token selects a row at most once. The bits equal `ScoresRows`.
    ScoresMap,
    /// The same output as `ExpertPlace`. The table holds `[rows,3]` as there, then `[rows]` slots in
    /// expert order: word `earlier+at` names the output row of that slot (`place_table`). A device
    /// tile moves the units through shared memory with it.
    ExpertPlaceTile,
    /// Output as `ScoresRows`. The expanded rows are packed two BF16 patterns a word (feature 2i low
    /// half, 2i+1 high half), so the second input holds half the words. Each widened operand is the
    /// exact f32 the unpacked row held. The bits equal `ScoresRows` on the unpacked rows.
    ScoresRowsPacked,
    /// Output as `ValuesRows` on the same packed rows. The bits equal `ValuesRows`.
    ValuesRowsPacked,
    /// Output as `ProjectRows` with BF16 weights packed two to a word: half the weight
    /// words, each half widened to the exact f32 the FP32 copy held. Bits equal `ProjectRows`.
    ProjectRowsBf16,
    /// Output as `ScoresMap` on the packed rows of `ScoresRowsPacked`: the tile kernel widens
    /// each packed half on staging, so its chains and bits equal `ScoresRowsPacked`.
    ScoresMapPacked,
}
#[repr(C)]
#[derive(Debug,Clone,Copy)]
pub struct Shape {
    pub op:Op,pub tokens:u32,pub heads:u32,pub dim:u32,pub taps:u32,pub rows:u32,pub cols:u32,
    /// Causal context of token zero. Later tokens add their index.
    pub first:u32,pub slots:u32,pub inner:CausalShape,
}
fn empty()->CausalShape {CausalShape {op:Causal::Project,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}}
impl Shape {
    pub fn new(op:Op,tokens:usize)->Option<Self> {
        Some(Self {op,tokens:u32::try_from(tokens).ok()?,heads:0,dim:0,taps:0,rows:0,cols:0,first:0,slots:0,inner:empty()})
    }
    pub fn wrapped(inner:CausalShape,tokens:usize)->Option<Self> {let mut s=Self::new(Op::Wrapped,tokens)?;s.inner=inner;Some(s)}
    /// ValuesRows threads for each output unit, when its value rows allow line reads.
    /// Each thread holds eight channels in registers. Every channel keeps its slot order.
    pub fn value_blocks(self)->Option<usize> {((self.op==Op::ValuesRows && self.dim%4==0 || self.op==Op::ValuesRowsPacked && self.dim%8==0) && self.taps%8==0 && self.taps>0).then_some(self.taps as usize/8)}
    /// ScoresRows threads for each output word when keys allow line reads: one thread for
    /// each token, head and slot, each its own chain, so a warp shares one key's cache lines.
    /// Packed keys take eight features a line (four packed words), so their dim is a multiple of 8
    /// and each head segment is a whole number of lines ((dim+taps) a multiple of 8), which keeps
    /// every head's first packed word 16-byte aligned like the row base the launch checks.
    pub fn score_lines(self)->bool {self.op==Op::ScoresRows && self.dim%4==0 && self.dim>0 && self.taps%4==0 && self.cols>0
        || self.op==Op::ScoresRowsPacked && self.dim%8==0 && self.dim>0 && (self.dim+self.taps)%8==0 && self.cols>0}
    /// Threads one launch needs for this many output units. `lines`: the kernel's line path
    /// runs, because query and key bases are 16-byte aligned. A score table then takes one
    /// thread for each entry and head: the kernel sets `head=thread/entries`, and a thread
    /// past that traps, so a table shorter than its tokens must not launch one per token.
    /// Without line reads the kernel computes one unit a thread, which needs every unit.
    pub fn threads(self,units:usize,lines:bool)->Option<usize> {
        if self.score_lines() && self.first>0 {
            let entries=self.first as usize;
            return (if lines {entries} else {entries.max(units/self.heads.max(1) as usize)}).checked_mul(self.heads as usize);
        }
        units.checked_mul(if self.score_lines() {self.cols as usize} else {self.value_blocks().unwrap_or(1)})
    }
    /// Complete input extents, weight extent, thread units and words for each unit.
    pub fn lengths(self)->Option<([usize;5],usize,usize,usize)> {
        let (m,h,d,t,r,k)=(self.tokens as usize,self.heads as usize,self.dim as usize,self.taps as usize,self.rows as usize,self.cols as usize);
        if m==0 || m>MAX_ROWS {return None;}
        let width=h.checked_mul(d)?;
        let slots=self.slots as usize;
        let result=match self.op {
            Op::Wrapped=>{let (x,w,units,words)=self.inner.lengths()?;
                if !matches!(self.inner.op,Causal::Prepare|Causal::GatedNorm|Causal::KeyNorm|Causal::Route|Causal::ExpertFold|Causal::StreamMean) {return None;}
                ([m.checked_mul(x[0])?,m.checked_mul(x[1])?,m.checked_mul(x[2])?,m.checked_mul(x[3])?,0],w,m.checked_mul(units)?,words)},
            Op::ProjectRows if r>0 && k>0=>([m.checked_mul(k)?,0,0,0,0],r.checked_mul(k)?,r.checked_mul(m)?,1),
            Op::ProjectRowsBf16 if r>0 && k>0 && k%2==0=>([m.checked_mul(k)?,0,0,0,0],r.checked_mul(k)?/2,r.checked_mul(m)?,1),
            Op::Transpose if r>0 && k>0=>([r.checked_mul(k)?,0,0,0,0],0,r.checked_mul(k)?,1),
            Op::NormApply if d>0=>([m.checked_mul(d)?,0,0,0,0],d,m.checked_mul(d)?,1),
            Op::ConvRows if width>0 && t>0=>{let row=m.checked_mul(width)?;([row,row,row,3usize.checked_mul(t-1)?.checked_mul(width)?,0],3usize.checked_mul(width)?.checked_mul(t)?,row.checked_mul(3)?,1)},
            Op::HistoryRows if width>0 && t>0=>{let row=m.checked_mul(width)?;([row,row,row,3usize.checked_mul(t-1)?.checked_mul(width)?,0],0,3usize.checked_mul(t-1)?.checked_mul(width)?,1)},
            Op::StepDecay if width>0 && d<=1024=>([m.checked_mul(h)?.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,0,0,0,0],0,m.checked_mul(width)?,1),
            Op::StepRows if width>0 && d<=1024=>([m.checked_mul(h)?.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,width.checked_mul(d)?,m.checked_mul(width)?,0,0],0,width,d.checked_add(m)?),
            Op::PoolScoresRows if width>0 && r>0 && t>0=>([m.checked_mul(width)?,r.checked_mul(d)?,m.checked_mul(h)?,0,0],0,m.checked_mul(r)?,1),
            Op::SelectRows if d>0 && t>0 && t%d==0 && k<=1 && slots>0=>([m.checked_mul(r)?,0,0,0,0],0,m,slots.checked_add(1)?),
            Op::ScoresRows if h>0 && d>0 && t>0 && k>0 && slots>0=>{let row=h.checked_mul(d.checked_add(t)?)?;
                // `first` counts entries of an optional (row, token, slot) table sorted by row.
                ([m.checked_mul(width)?,r.checked_mul(row)?,m.checked_mul(slots.checked_add(1)?)?,(self.first as usize).checked_mul(3)?,0],0,m.checked_mul(h)?,k)},
            Op::ScoresMap if h>0 && d>0 && t>0 && k>0 && slots>0 && r>0=>{let row=h.checked_mul(d.checked_add(t)?)?;
                ([m.checked_mul(width)?,r.checked_mul(row)?,m.checked_mul(slots.checked_add(1)?)?,r.checked_mul(m)?,0],0,m.checked_mul(h)?,k)},
            Op::SoftmaxRows if h>0 && k>0 && slots>0=>([m.checked_mul(h)?.checked_mul(k)?,m.checked_mul(slots.checked_add(1)?)?,0,0,0],0,m.checked_mul(h)?,k),
            Op::ValuesRows if h>0 && d>0 && t>0 && k>0 && slots>0=>{let row=h.checked_mul(d.checked_add(t)?)?;
                ([m.checked_mul(h)?.checked_mul(k)?,r.checked_mul(row)?,m.checked_mul(slots.checked_add(1)?)?,0,0],0,m.checked_mul(h)?,t)},
            Op::ScoresRowsPacked if h>0 && d>0 && d%2==0 && t>0 && t%2==0 && k>0 && slots>0=>{let row=h.checked_mul(d.checked_add(t)?)?/2;
                ([m.checked_mul(width)?,r.checked_mul(row)?,m.checked_mul(slots.checked_add(1)?)?,(self.first as usize).checked_mul(3)?,0],0,m.checked_mul(h)?,k)},
            Op::ScoresMapPacked if h>0 && d>0 && d%2==0 && t>0 && t%2==0 && k>0 && slots>0 && r>0=>{let row=h.checked_mul(d.checked_add(t)?)?/2;
                ([m.checked_mul(width)?,r.checked_mul(row)?,m.checked_mul(slots.checked_add(1)?)?,r.checked_mul(m)?,0],0,m.checked_mul(h)?,k)},
            Op::ValuesRowsPacked if h>0 && d>0 && d%2==0 && t>0 && t%2==0 && k>0 && slots>0=>{let row=h.checked_mul(d.checked_add(t)?)?/2;
                ([m.checked_mul(h)?.checked_mul(k)?,r.checked_mul(row)?,m.checked_mul(slots.checked_add(1)?)?,0,0],0,m.checked_mul(h)?,t)},
            Op::PermuteRows if r>0 && k>0=>(([(self.first as usize).checked_mul(k)?,r,0,0,0]),0,r.checked_mul(k)?,1),
            Op::ExpertPlace if r>0 && k>0 && d>0 && t>0 && k==d.checked_mul(t)? && slots>0=>([slots.checked_mul(t)?,r.checked_mul(3)?,0,0,0],0,r.checked_mul(k)?,1),
            Op::ExpertPlaceTile if r>0 && k>0 && d>0 && t>0 && k==d.checked_mul(t)? && slots>0=>([slots.checked_mul(t)?,r.checked_mul(4)?,0,0,0],0,r.checked_mul(k)?,1),
            _=>return None,
        };
        if result.0.into_iter().chain([result.1,result.2.checked_mul(result.3)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
        Some(result)
    }
}
/// The host checks every extent once. Threads read the checked values from here.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Arguments {pub input:[u64;5],pub lengths:[u64;5],pub weight:u64,pub weights:u64,pub output:u64,pub units:u64,pub words:u64,pub shape:Shape}
/// Causal history is oldest first: old history, then each earlier token's row.
#[inline(always)]
fn history_value(plane:&[f32],old:&[f32],position:usize,keep:usize,width:usize,ch:usize)->f32 {
    if position<keep {old[position*width+ch]} else {plane[(position-keep)*width+ch]}
}
#[inline(always)]
pub(crate) fn select_all(s:Shape,token:usize,scores:&[f32],out:&mut[f32])->bool {
    // Every complete pool is kept, so the selected IDs are ascending and
    // independent of score values. A NaN score has no defined order: refuse it.
    let d=s.dim as usize;let context=s.first as usize+token;let groups=context/d;let slots=s.slots as usize;
    let mut valid=true;for &score in &scores[..groups] {valid&=score==score;}
    let mut count=groups*d;for id in 0..count {out[id]=id as f32;}
    if s.cols!=0 {for id in count..context {out[count]=id as f32;count+=1;}}
    for v in &mut out[count..slots] {*v=-1.0;}
    out[slots]=if valid {count as f32} else {-2.0};valid
}
/// Top pools by descending score, then ascending pool ID. This is the scalar
/// selection's total order without its repeated scans. Sorted IDs then expand
/// backward as before. A NaN score has no defined order: refuse it.
pub(crate) fn select_top(s:Shape,token:usize,scores:&[f32],out:&mut[f32]) {
    let d=s.dim as usize;let context=s.first as usize+token;let groups=context/d;let take=s.taps as usize/d;let slots=s.slots as usize;
    let worse=|a:usize,b:usize|scores[a]<scores[b] || scores[a]==scores[b] && a>b;
    let mut valid=true;for &score in &scores[..groups] {valid&=score==score;}
    // A heap of the best pools so far with the worst pool at its root.
    fn sift(out:&mut[f32],len:usize,mut at:usize,before:&impl Fn(usize,usize)->bool) {
        loop {let mut child=2*at+1;if child>=len {return;}
            if child+1<len && before(out[child+1] as usize,out[child] as usize) {child+=1;}
            if !before(out[child] as usize,out[at] as usize) {return;}
            let held=out[at];out[at]=out[child];out[child]=held;at=child;}
    }
    for group in 0..take {out[group]=group as f32;}
    let mut at=take/2;while at>0 {at-=1;sift(out,take,at,&worse);}
    for group in take..groups {if worse(out[0] as usize,group) {out[0]=group as f32;sift(out,take,0,&worse);}}
    // Ascending IDs: a heap sort on the same slots.
    let larger=|a:usize,b:usize|a>b;
    let mut at=take/2;while at>0 {at-=1;sift(out,take,at,&larger);}
    let mut end=take;while end>1 {end-=1;let held=out[0];out[0]=out[end];out[end]=held;sift(out,end,0,&larger);}
    for i in (0..take).rev() {let group=out[i] as usize;for slot in 0..d {out[i*d+slot]=(group*d+slot) as f32;}}
    let mut count=take*d;if s.cols!=0 {for id in groups*d..context {out[count]=id as f32;count+=1;}}
    for v in &mut out[count..slots] {*v=-1.0;}
    out[slots]=if valid {count as f32} else {-2.0};
}
/// One value column's words inside the shared StepRows output.
pub trait StepStore {fn get(&self,key:usize)->f32;fn set(&mut self,key:usize,value:f32);fn output(&mut self,token:usize,value:f32);}
/// The key folds and value columns follow StepColumns for each token. Each fold
/// keeps its own accumulator and ascending keys, so joining its loop with the state
/// update changes no value. Only token order carries state.
#[inline(always)]
pub fn step_column<S:StepStore>(s:Shape,index:usize,x:[&[f32];5],store:&mut S) {
    let (m,h,d)=(s.tokens as usize,s.heads as usize,s.dim as usize);
    let head=index/d;let value=index%d;let stride=4*d+1;
    let state=&x[1][head*d*d..(head+1)*d*d];
    for key in 0..d {store.set(key,state[key*d+value]);}
    for token in 0..m {
        let at=token*h+head;let p=&x[0][at*stride..(at+1)*stride];let decay=&x[2][at*d..(at+1)*d];
        let (keys,rest)=p.split_at(d);let (update,rest)=rest.split_at(d);let target=rest[value];let beta=rest[2*d];
        let mut prediction=0.0;for key in 0..d {let v=store.get(key)*decay[key];store.set(key,v);prediction+=v*update[key];}
        let delta=(target-prediction)*beta;
        let mut sum=0.0;for key in 0..d {let v=store.get(key)+update[key]*delta;store.set(key,v);sum+=v*keys[key];}store.output(token,round(sum));
    }
}
/// Threads of one warp read the same latent rows when they share a head. This
/// orders threads only. Each output keeps its own fold.
#[inline(always)]
pub fn thread_unit(s:Shape,thread:usize)->usize {
    if matches!(s.op,Op::ScoresRows|Op::ScoresMap|Op::ValuesRows|Op::ScoresRowsPacked|Op::ValuesRowsPacked|Op::ScoresMapPacked) {let m=s.tokens as usize;(thread%m)*s.heads as usize+thread/m} else {thread}
}
/// Transpose and ExpertPlace read an exchange buffer whole. With `first` set they refuse a
/// non-finite word there, so the host need not scan the mapped buffer. A refusal traps.
#[inline(always)]
fn refuse_non_finite(s:Shape,value:f32) {if s.first!=0 && value.to_bits()&0x7f80_0000==0x7f80_0000 {panic!("exchanged value is not finite");}}
/// One feature of a packed expanded row: word `base+feature/2`, low half for even features, high half
/// for odd ones, widened by the shift that made the unpacked row's f32 from the same BF16 pattern.
#[inline(always)]
pub(crate) fn packed_feature(x:&[f32],base:usize,feature:usize)->f32 {
    let bits=x[base+feature/2].to_bits();if feature%2==0 {f32::from_bits(bits<<16)} else {f32::from_bits(bits&0xffff_0000)}
}
/// Write one complete disjoint output unit. The host validates all extents.
pub fn unit(s:Shape,index:usize,x:[&[f32];5],w:&[f32],out:&mut[f32]) {
    let (m,h,d,t,r,k)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);let width=h*d;
    match s.op {
        Op::Wrapped=>{
            let unit_count=x_units(s);let token=index/unit_count;
            let part=|n:usize|{let size=x[n].len()/m;&x[n][token*size..(token+1)*size]};
            let (a,b,c,e)=(part(0),part(1),part(2),part(3));
            causal_ops::unit(s.inner,index%unit_count,[a,b,c,e],w,out);
        },
        Op::ProjectRows=>{let row=index/m;let token=index%m;let mut sum=0.0;
            let input=&x[0][token*k..(token+1)*k];let weight=&w[row*k..(row+1)*k];
            for j in 0..k {sum+=input[j]*weight[j];}out[0]=sum;},
        Op::ProjectRowsBf16=>{let row=index/m;let token=index%m;let mut sum=0.0;
            let input=&x[0][token*k..(token+1)*k];
            for j in 0..k {let bits=w[(row*k+j)/2].to_bits();let weight=if j%2==0 {f32::from_bits(bits<<16)} else {f32::from_bits(bits&0xffff_0000)};sum+=input[j]*weight;}out[0]=sum;},
        Op::Transpose=>{out[0]=x[0][(index%r)*k+index/r];refuse_non_finite(s,out[0]);},
        Op::NormApply=>{let v=x[0][index]*w[index%d];out[0]=if k!=0 {round(v)} else {v};},
        Op::ConvRows=>{
            let plane=(index/width)%3;let token=index/(3*width);let ch=index%width;let history=(t-1)*width;
            let old=&x[3][plane*history..(plane+1)*history];let keep=t-1;let mut sum=0.0;
            for tap in 0..t {sum+=history_value(x[plane],old,tap+token,keep,width,ch)*w[(plane*width+ch)*t+tap];}
            out[0]=round(sum*sigmoid(sum));
        },
        Op::HistoryRows=>{
            let history=(t-1)*width;let plane=index/history;let lane=index%history;let tap=lane/width;let ch=lane%width;
            let old=&x[3][plane*history..(plane+1)*history];
            out[0]=history_value(x[plane],old,tap+m,t-1,width,ch);
        },
        Op::StepDecay=>{let key=index%d;out[0]=exp(x[0][(index/d)*(4*d+1)+3*d+key]);},
        // The kernel writes these columns through StepStore. See `step_column`.
        Op::StepRows=>{},
        Op::PoolScoresRows=>{
            let token=index/r;let pool=index%r;let pools=(s.first as usize+token)/t;let mut score=0.0;
            if pool<pools {let query=&x[0][token*width..(token+1)*width];let key=&x[1][pool*d..(pool+1)*d];let weights=&x[2][token*h..(token+1)*h];
                for head in 0..h {let mut dot=0.0;for j in 0..d {dot+=query[head*d+j]*key[j];}
                    score+=(weights[head]*(1.0/root(h as f32)))*(dot*(1.0/root(d as f32))).max(0.0);}}
            out[0]=score;
        },
        Op::SelectRows=>{
            let token=index;let context=s.first as usize+token;let groups=context/d;let row=&x[0][token*r..(token+1)*r];
            if t/d>=groups {select_all(s,token,row,out);} else {select_top(s,token,row,out);}
        },
        Op::ScoresRows|Op::ScoresMap=>{
            let token=index/h;let head=index%h;let query=&x[0][index*d..(index+1)*d];let row=h*(d+t);
            let selected=&x[2][token*(s.slots as usize+1)..(token+1)*(s.slots as usize+1)];let count=selected[s.slots as usize] as usize;
            let scale=1.0/root(d as f32);for v in out.iter_mut() {*v=0.0;}
            for slot in 0..count {let id=selected[slot] as usize;let key=&x[1][id*row+head*(d+t)..id*row+head*(d+t)+d];let mut sum=0.0;
                for j in 0..d {sum+=query[j]*key[j];}out[slot]=round(round(sum)*scale);}
        },
        Op::ScoresRowsPacked|Op::ScoresMapPacked=>{
            let token=index/h;let head=index%h;let query=&x[0][index*d..(index+1)*d];let row=h*(d+t)/2;let segment=(d+t)/2;
            let selected=&x[2][token*(s.slots as usize+1)..(token+1)*(s.slots as usize+1)];let count=selected[s.slots as usize] as usize;
            let scale=1.0/root(d as f32);for v in out.iter_mut() {*v=0.0;}
            for slot in 0..count {let id=selected[slot] as usize;let base=id*row+head*segment;let mut sum=0.0;
                for j in 0..d {sum+=query[j]*packed_feature(x[1],base,j);}out[slot]=round(round(sum)*scale);}
        },
        Op::SoftmaxRows=>{
            let token=index/h;let selected=&x[1][token*(s.slots as usize+1)..(token+1)*(s.slots as usize+1)];
            let count=selected[s.slots as usize] as usize;for v in out.iter_mut() {*v=0.0;}if count==0 {return;}
            let row=&x[0][index*k..(index+1)*k];
            let mut maximum=f32::NEG_INFINITY;for &v in &row[..count] {maximum=maximum.max(v);}
            let mut denominator=0.0;for &v in &row[..count] {denominator+=exp(v-maximum);}
            for slot in 0..count {out[slot]=round(exp(row[slot]-maximum)/denominator);}
        },
        Op::ValuesRows=>{
            // Each output keeps ascending slots. Visiting one slot's complete
            // value row at a time changes no output's order of additions.
            let token=index/h;let head=index%h;let selected=&x[2][token*(s.slots as usize+1)..(token+1)*(s.slots as usize+1)];
            let count=selected[s.slots as usize] as usize;let row=h*(d+t);let p=&x[0][index*k..(index+1)*k];
            for v in out.iter_mut() {*v=0.0;}
            for slot in 0..count {let id=selected[slot] as usize;let weight=p[slot];let value=&x[1][id*row+head*(d+t)+d..id*row+head*(d+t)+d+t];
                for j in 0..t {out[j]+=weight*value[j];}}
            for v in out.iter_mut() {*v=round(*v);}
        },
        Op::ValuesRowsPacked=>{
            let token=index/h;let head=index%h;let selected=&x[2][token*(s.slots as usize+1)..(token+1)*(s.slots as usize+1)];
            let count=selected[s.slots as usize] as usize;let row=h*(d+t)/2;let segment=(d+t)/2;let p=&x[0][index*k..(index+1)*k];
            for v in out.iter_mut() {*v=0.0;}
            for slot in 0..count {let id=selected[slot] as usize;let weight=p[slot];let base=id*row+head*segment;
                for j in 0..t {out[j]+=weight*packed_feature(x[1],base,d+j);}}
            for v in out.iter_mut() {*v=round(*v);}
        },
        Op::PermuteRows=>{let row=index/k;let lane=index%k;out[0]=x[0][x[1][row] as usize*k+lane];},
        Op::ExpertPlace|Op::ExpertPlaceTile=>{let row=index/k;let lane=index%k;let entry=&x[1][row*3..row*3+3];
            let earlier=entry[0] as usize;let tokens=entry[1] as usize;let at=entry[2] as usize;
            out[0]=x[0][(lane/d)*s.slots as usize+earlier*d+(lane%d)*tokens+at];refuse_non_finite(s,out[0]);},
    }
}
#[inline(always)]
fn x_units(s:Shape)->usize {match s.inner.lengths() {Some((_,_,units,_))=>units,None=>1}}
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(s:Shape,x:[&[f32];5],w:&[f32])->Result<Vec<f32>,String> {
    let (lengths,weight,units,words)=s.lengths().ok_or("invalid prompt row shape")?;
    if x.iter().zip(lengths).any(|(v,n)|v.len()!=n) || w.len()!=weight {return Err(format!("prompt row {:?} operand extent differs: {:?} wanted {lengths:?}",s.op,x.map(<[f32]>::len)));}
    let mut out=vec![0.0;units*words];
    if s.op==Op::StepRows {
        struct Host<'a> {out:&'a mut [f32],head:usize,value:usize,d:usize,width:usize,states:usize}
        impl StepStore for Host<'_> {
            fn get(&self,key:usize)->f32 {self.out[(self.head*self.d+key)*self.d+self.value]}
            fn set(&mut self,key:usize,value:f32) {self.out[(self.head*self.d+key)*self.d+self.value]=value;}
            fn output(&mut self,token:usize,value:f32) {self.out[self.states+token*self.width+self.head*self.d+self.value]=value;}
        }
        let d=s.dim as usize;let width=s.heads as usize*d;
        for i in 0..units {let mut store=Host {out:&mut out,head:i/d,value:i%d,d,width,states:width*d};step_column(s,i,x,&mut store);}
        return Ok(out);
    }
    for (i,row) in out.chunks_exact_mut(words).enumerate() {unit(s,i,x,w,row);}Ok(out)
}
/// The table of `ExpertPlaceTile`: the `[rows,3]` table of `ExpertPlace`, then for every slot (`earlier+at`) its output
/// row. `None` unless every entry is an exact small integer, `at < tokens`, and the slots are a permutation of
/// `0..rows` (the place table of a real expert grouping always is); the caller then keeps `ExpertPlace`.
#[cfg(not(target_arch="nvptx64"))]
pub fn place_table(places:&[f32])->Option<Vec<f32>> {
    if places.is_empty() || places.len()%3!=0 {return None;}
    let rows=places.len()/3;if rows>=(1<<24) {return None;}
    let whole=|v:f32|(v>=0.0 && v<16777216.0 && v==v.trunc()).then_some(v as usize);
    let mut slot_row=vec![u32::MAX;rows];
    for (row,e) in places.chunks_exact(3).enumerate() {
        let (earlier,tokens,at)=(whole(e[0])?,whole(e[1])?,whole(e[2])?);
        if at>=tokens {return None;}
        let slot=earlier+at;if slot>=rows || slot_row[slot]!=u32::MAX {return None;}slot_row[slot]=row as u32;
    }
    let mut table=places.to_vec();table.extend(slot_row.into_iter().map(|row| row as f32));Some(table)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_ops::reference as causal;
    fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
    fn value(seed:usize,scale:f32)->f32 {if seed%19==0 {-0.0} else {((seed*31+7)%53) as f32*scale-26.0*scale}}
    fn rows(tokens:usize)->Shape {Shape::new(Op::Transpose,tokens).unwrap()}
    #[test]
    fn plain_rows_match_each_scalar_projection() {
        for m in [1usize,2,5,33] {for k in [1usize,17,128] {
            let s=Shape {op:Op::ProjectRows,rows:7,cols:k as u32,..rows(m)};
            let x=(0..m*k).map(|i|value(i,0.031)).collect::<Vec<_>>();let w=(0..7*k).map(|i|value(i+3,0.017)).collect::<Vec<_>>();
            let got=reference(s,[&x,&[],&[],&[],&[]],&w).unwrap();
            for token in 0..m {let scalar=CausalShape {op:Causal::Project,heads:0,dim:0,taps:0,rows:7,cols:k as u32,epsilon:0.0,bound:0.0};
                let expected=causal(scalar,[&x[token*k..(token+1)*k],&[],&[],&[]],&w).unwrap();
                for row in 0..7 {assert_eq!(got[row*m+token].to_bits(),expected[row].to_bits());}}
        }}
    }
    #[test]
    fn transpose_moves_bits_only() {
        let s=Shape {op:Op::Transpose,rows:3,cols:5,..rows(1)};let x=(0..15).map(|i|value(i,0.1)).collect::<Vec<_>>();
        let got=reference(s,[&x,&[],&[],&[],&[]],&[]).unwrap();
        for a in 0..3 {for b in 0..5 {assert_eq!(got[b*3+a].to_bits(),x[a*5+b].to_bits());}}
    }
    #[test]
    fn convolution_rows_carry_history_exactly_as_token_order() {
        for m in [1usize,2,3,4,7] {for taps in [1usize,2,4] {
            let (h,d)=(3usize,2usize);let width=h*d;
            let s=Shape {op:Op::ConvRows,heads:h as u32,dim:d as u32,taps:taps as u32,..rows(m)};
            let row=|plane:usize|(0..m*width).map(|i|value(i+plane*101,0.0625)).collect::<Vec<_>>();let (q,k,v)=(row(0),row(1),row(2));
            let old=(0..3*(taps-1)*width).map(|i|value(i+11,0.03125)).collect::<Vec<_>>();
            let w=(0..3*width*taps).map(|i|value(i+5,0.0078125)).collect::<Vec<_>>();
            let got=reference(s,[&q,&k,&v,&old,&[]],&w).unwrap();
            let next=reference(Shape {op:Op::HistoryRows,..s},[&q,&k,&v,&old,&[]],&[]).unwrap();
            let mut state=old.clone();let scalar=CausalShape {op:Causal::Conv,heads:h as u32,dim:d as u32,taps:taps as u32,rows:0,cols:0,epsilon:0.0,bound:0.0};
            for token in 0..m {
                let packed=[&q[token*width..(token+1)*width],&k[token*width..(token+1)*width],&v[token*width..(token+1)*width]].concat();
                let out=causal(scalar,[&packed,&state,&[],&[]],&w).unwrap();
                let mut c=Vec::new();let mut hist=Vec::new();for p in 0..3 {c.extend_from_slice(&out[p*width*taps..p*width*taps+width]);hist.extend_from_slice(&out[p*width*taps+width..(p+1)*width*taps]);}
                assert_eq!(bits(&c),bits(&got[token*3*width..(token+1)*3*width]),"m={m} taps={taps} token={token}");state=hist;
            }
            assert_eq!(bits(&next),bits(&state),"m={m} taps={taps}");
        }}
    }
    #[test]
    fn recurrence_rows_match_each_scalar_step_in_token_order() {
        for m in [1usize,2,3,9] {for d in [1usize,2,8] {
            let h=3usize;let stride=4*d+1;
            let s=Shape {op:Op::StepRows,heads:h as u32,dim:d as u32,..rows(m)};
            let p=(0..m*h*stride).map(|i|value(i,0.03125)).collect::<Vec<_>>();let old=(0..h*d*d).map(|i|value(i+9,0.015625)).collect::<Vec<_>>();
            let decay=reference(Shape {op:Op::StepDecay,..s},[&p,&[],&[],&[],&[]],&[]).unwrap();
            let all=reference(s,[&p,&old,&decay,&[],&[]],&[]).unwrap();let (state,outputs)=all.split_at(h*d*d);
            let mut scalar=old.clone();let one=CausalShape {op:Causal::Step,heads:h as u32,dim:d as u32,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
            for token in 0..m {let out=causal(one,[&p[token*h*stride..(token+1)*h*stride],&scalar,&[],&[]],&[]).unwrap();
                let mut next=Vec::new();let mut recurrent=Vec::new();for head in 0..h {next.extend_from_slice(&out[head*(d*d+d)..head*(d*d+d)+d*d]);recurrent.extend_from_slice(&out[head*(d*d+d)+d*d..(head+1)*(d*d+d)]);}
                assert_eq!(bits(&recurrent),bits(&outputs[token*h*d..(token+1)*h*d]),"m={m} d={d} token={token}");scalar=next;}
            assert_eq!(bits(state),bits(&scalar),"m={m} d={d}");
        }}
    }
    #[test]
    fn wrapped_rows_equal_each_scalar_unit() {
        let inner=CausalShape {op:Causal::StreamMean,heads:4,dim:8,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
        let x=(0..3*32).map(|i|value(i,0.25)).collect::<Vec<_>>();
        let got=reference(Shape::wrapped(inner,3).unwrap(),[&x,&[],&[],&[],&[]],&[]).unwrap();
        for token in 0..3 {let one=causal(inner,[&x[token*32..(token+1)*32],&[],&[],&[]],&[]).unwrap();assert_eq!(bits(&one),bits(&got[token*8..(token+1)*8]));}
        let gated=CausalShape {op:Causal::GatedNorm,heads:2,dim:4,taps:0,rows:0,cols:0,epsilon:1e-5,bound:0.0};
        let (a,b)=((0..3*8).map(|i|value(i,0.25)).collect::<Vec<_>>(),(0..3*8).map(|i|value(i+2,0.125)).collect::<Vec<_>>());let w=[0.5,1.5,0.25,0.75];
        let got=reference(Shape::wrapped(gated,3).unwrap(),[&a,&b,&[],&[],&[]],&w).unwrap();
        for token in 0..3 {let one=causal(gated,[&a[token*8..(token+1)*8],&b[token*8..(token+1)*8],&[],&[]],&w).unwrap();assert_eq!(bits(&one),bits(&got[token*8..(token+1)*8]));}
    }
    #[test]
    fn project_rows_bf16_equals_project_rows_on_widened_weights() {
        for (r,cols,m) in [(7usize,128usize,4usize),(16384,512,2),(2048,1536,3),(4096,128,5)] {
            let weight=(0..r*cols).map(|i|round(value(i,0.03125))).collect::<Vec<_>>();
            let packed=weight.chunks_exact(2).map(|p|f32::from_bits((p[0].to_bits()>>16)|(p[1].to_bits()&0xffff_0000))).collect::<Vec<_>>();
            let input=(0..m*cols).map(|i|value(i+1,0.0625)).collect::<Vec<_>>();
            let plain=Shape {op:Op::ProjectRows,rows:r as u32,cols:cols as u32,..rows(m)};
            let got=reference(plain,[&input,&[],&[],&[],&[]],&weight).unwrap();
            let bf16_form=reference(Shape {op:Op::ProjectRowsBf16,..plain},[&input,&[],&[],&[],&[]],&packed).unwrap();
            assert_eq!(bits(&got),bits(&bf16_form),"{r}x{cols}x{m}");
        }
    }
    #[test]
    fn packed_attention_rows_equal_the_unpacked_rows() {
        // Even dims, unequal live prefixes, a repeated row and reordered slots. Expanded words are
        // BF16-rounded first, as every producer leaves them; packing must then change no output bit.
        let (h,kd,vd)=(3usize,8usize,6usize);let (first,m)=(3usize,5usize);let total=first+m;let row=h*(kd+vd);let slots=6usize;let cap=slots.min(total);
        let e=(0..total*row).map(|i|round(value(i,0.0625))).collect::<Vec<_>>();
        assert!(e.iter().all(|v|v.to_bits()&0xffff==0));
        let packed=e.chunks_exact(2).map(|pair|f32::from_bits((pair[0].to_bits()>>16)|(pair[1].to_bits()&0xffff_0000))).collect::<Vec<_>>();
        let q=(0..m*h*kd).map(|i|value(i+3,0.125)).collect::<Vec<_>>();
        let mut selected=vec![-1.0f32;m*(slots+1)];
        for token in 0..m {let context=first+token;let count=context.min(slots);for slot in 0..count {selected[token*(slots+1)+slot]=(((context-1-slot)*(slot+1))%context) as f32;}selected[token*(slots+1)+slots]=count as f32;}
        let base=Shape {heads:h as u32,taps:vd as u32,rows:total as u32,cols:cap as u32,slots:slots as u32,..rows(m)};
        let scores=reference(Shape {op:Op::ScoresRows,dim:kd as u32,..base},[&q,&e,&selected,&[],&[]],&[]).unwrap();
        let scores_packed=reference(Shape {op:Op::ScoresRowsPacked,dim:kd as u32,..base},[&q,&packed,&selected,&[],&[]],&[]).unwrap();
        assert_eq!(bits(&scores),bits(&scores_packed));
        let prob=reference(Shape {op:Op::SoftmaxRows,..base},[&scores,&selected,&[],&[],&[]],&[]).unwrap();
        let values=reference(Shape {op:Op::ValuesRows,dim:kd as u32,..base},[&prob,&e,&selected,&[],&[]],&[]).unwrap();
        let values_packed=reference(Shape {op:Op::ValuesRowsPacked,dim:kd as u32,..base},[&prob,&packed,&selected,&[],&[]],&[]).unwrap();
        assert_eq!(bits(&values),bits(&values_packed));
        // Odd dims have no packed form.
        assert!(Shape {op:Op::ScoresRowsPacked,dim:3,..base}.lengths().is_none());
        assert!(Shape {op:Op::ValuesRowsPacked,taps:5,..base}.lengths().is_none());
    }
    #[test]
    fn selection_rows_match_the_scalar_selection_at_every_context() {
        for tail in [0u32,1] {for first in 1usize..90 {
            let (d,budget)=(4usize,16usize);let slots=budget+if tail!=0 {d-1} else {0};let m=5;let r=(first+m-1)/d;if r==0 {continue;}
            let scores=(0..m*r).map(|i|if i%5==1 {f32::NEG_INFINITY} else if i%7==2 {0.5} else {value(i,0.125)}).collect::<Vec<_>>();
            let s=Shape {op:Op::SelectRows,dim:d as u32,taps:budget as u32,rows:r as u32,cols:tail,first:first as u32,slots:slots as u32,..rows(m)};
            let got=reference(s,[&scores,&[],&[],&[],&[]],&[]).unwrap();
            for token in 0..m {let context=first+token;
                let one=CausalShape {op:Causal::Select,heads:0,dim:d as u32,taps:budget as u32,rows:context as u32,cols:tail,epsilon:0.0,bound:0.0};
                let mut expected=vec![0.0;slots+1];crate::latent_ops::unit(one,0,[&scores[token*r..token*r+context/d],&[],&[],&[]],&[],&mut expected);
                assert_eq!(bits(&got[token*(slots+1)..(token+1)*(slots+1)]),bits(&expected),"first={first} token={token} tail={tail}");
            }
        }}
        // Selection order has no meaning for a NaN score. The caller must refuse it.
        let s=Shape {op:Op::SelectRows,dim:4,taps:16,rows:1,cols:1,first:4,slots:19,..rows(1)};
        assert_eq!(reference(s,[&[f32::NAN],&[],&[],&[],&[]],&[]).unwrap()[19],-2.0);
    }
    #[test]
    fn attention_rows_match_each_scalar_fold_through_shared_latents() {
        let (h,kd,vd)=(2usize,3usize,2usize);let (first,m)=(2usize,6usize);let total=first+m;let row=h*(kd+vd);let slots=7usize;let cap=slots.min(total);
        let e=(0..total*row).map(|i|value(i,0.0625)).collect::<Vec<_>>();
        let q=(0..m*h*kd).map(|i|value(i+3,0.125)).collect::<Vec<_>>();
        // Choose unequal live prefixes, including a reordered slot.
        let mut selected=vec![-1.0f32;m*(slots+1)];
        for token in 0..m {let context=first+token;let count=context.min(slots);for slot in 0..count {selected[token*(slots+1)+slot]=(context-1-slot) as f32;}selected[token*(slots+1)+slots]=count as f32;}
        let base=Shape {heads:h as u32,taps:vd as u32,rows:total as u32,cols:cap as u32,slots:slots as u32,..rows(m)};
        let scores=reference(Shape {op:Op::ScoresRows,dim:kd as u32,..base},[&q,&e,&selected,&[],&[]],&[]).unwrap();
        let prob=reference(Shape {op:Op::SoftmaxRows,..base},[&scores,&selected,&[],&[],&[]],&[]).unwrap();
        let values=reference(Shape {op:Op::ValuesRows,dim:kd as u32,..base},[&prob,&e,&selected,&[],&[]],&[]).unwrap();
        for token in 0..m {let count=selected[token*(slots+1)+slots] as usize;
            let mut packed=Vec::new();for slot in 0..count {let id=selected[token*(slots+1)+slot] as usize;packed.extend_from_slice(&e[id*row..(id+1)*row]);}
            let unpack=CausalShape {op:Causal::UnpackKv,heads:h as u32,dim:kd as u32,taps:count as u32,rows:cap as u32,cols:vd as u32,epsilon:0.0,bound:0.0};
            let plain=causal(unpack,[&packed,&[],&[],&[]],&[]).unwrap();let (keys,vals)=plain.split_at(cap*h*kd);let n=[count as f32];
            let one=CausalShape {op:Causal::Scores,heads:h as u32,dim:kd as u32,taps:0,rows:cap as u32,cols:0,epsilon:0.0,bound:0.0};
            let a=causal(one,[&q[token*h*kd..(token+1)*h*kd],keys,&n,&[]],&[]).unwrap();
            assert_eq!(bits(&a),bits(&scores[token*h*cap..(token+1)*h*cap]),"scores token={token}");
            let b=causal(CausalShape {op:Causal::Softmax,..one},[&a,&n,&[],&[]],&[]).unwrap();
            assert_eq!(bits(&b),bits(&prob[token*h*cap..(token+1)*h*cap]),"softmax token={token}");
            let c=causal(CausalShape {op:Causal::Values,dim:vd as u32,..one},[&b,vals,&n,&[]],&[]).unwrap();
            assert_eq!(bits(&c),bits(&values[token*h*vd..(token+1)*h*vd]),"values token={token}");
        }
    }
    #[test]
    fn a_score_table_launches_one_thread_per_entry_and_head_and_every_word_matches() {
        // (tokens, heads, live slots of each token), 0 < entries < tokens first.
        for (m,h,counts) in [(2usize,1usize,vec![1usize,0]),(5,2,vec![2,0,1,0,0]),(4,3,vec![0,0,0,1]),(3,3,vec![1,1,1]),(2,2,vec![2,2])] {
            let (d,t,slots,total)=(4usize,4usize,2usize,6usize);let row=h*(d+t);
            let e=(0..total*row).map(|i|value(i,0.0625)).collect::<Vec<_>>();let q=(0..m*h*d).map(|i|value(i+3,0.125)).collect::<Vec<_>>();
            let mut selected=vec![-1.0f32;m*(slots+1)];let mut order=Vec::new();
            for (token,&count) in counts.iter().enumerate() {
                for slot in 0..count {let id=(token*2+slot)%total;selected[token*(slots+1)+slot]=id as f32;order.push((id,token,slot));}
                selected[token*(slots+1)+slots]=count as f32;
            }
            order.sort_unstable();let entries=order.len();
            let table=order.iter().flat_map(|o|[o.0 as f32,o.1 as f32,o.2 as f32]).collect::<Vec<_>>();
            let s=Shape {op:Op::ScoresRows,heads:h as u32,dim:d as u32,taps:t as u32,rows:total as u32,cols:slots as u32,slots:slots as u32,first:entries as u32,..rows(m)};
            assert!(s.score_lines());let units=m*h;
            // The old grid. It stays for tables at least as long as the tokens, so those launches do not change.
            let old=entries.max(m)*h;let lines=s.threads(units,true).unwrap();
            assert_eq!(lines,entries*h);assert_eq!(s.threads(units,false),Some(old));
            if entries>=m {assert_eq!(lines,old);} else {assert!((old-1)/entries>=h,"the old grid reaches the trap for m={m}");}
            // The kernel's line path, thread by thread. Output starts zeroed, and any trap is a failure.
            let mut out=vec![0.0f32;units*slots];
            for thread in 0..lines {
                let entry=&table[(thread%entries)*3..(thread%entries)*3+3];let (id,token,slot,head)=(entry[0] as usize,entry[1] as usize,entry[2] as usize,thread/entries);
                assert!(token<m && head<h,"trap at thread {thread} of m={m}");let live=&selected[token*(slots+1)..(token+1)*(slots+1)];
                assert!(slot<live[slots] as usize && live[slot] as usize==id,"trap at thread {thread} of m={m}");
                let index=token*h+head;let query=&q[index*d..(index+1)*d];let key=&e[id*row+head*(d+t)..id*row+head*(d+t)+d];
                let mut sum=0.0f32;for j in 0..d {sum+=query[j]*key[j];}
                out[index*slots+slot]=round(round(sum)*(1.0/root(d as f32)));
            }
            let want=reference(s,[&q,&e,&selected,&table,&[]],&[]).unwrap();assert_eq!(bits(&out),bits(&want),"m={m} h={h}");
            // Without line reads the kernel runs one unit for each thread below tokens*heads: all of them.
            let mut seen=vec![false;units];for thread in 0..s.threads(units,false).unwrap() {if thread/units==0 {seen[thread_unit(s,thread)]=true;}}
            assert!(seen.iter().all(|&v|v),"fallback misses a unit of m={m}");
        }
        // No table: one thread for each output word, as before.
        let plain=Shape {op:Op::ScoresRows,heads:2,dim:4,taps:4,rows:6,cols:2,slots:2,..rows(3)};
        assert_eq!(plain.threads(6,true),Some(12));assert_eq!(plain.threads(6,false),Some(12));
    }
    #[test]
    fn pool_scores_rows_match_each_scalar_prefix() {
        let (h,d,t)=(2usize,3usize,4usize);let (first,m)=(3usize,6usize);let r=(first+m-1)/t;
        let q=(0..m*h*d).map(|i|value(i,0.125)).collect::<Vec<_>>();let k=(0..r*d).map(|i|value(i+1,0.0625)).collect::<Vec<_>>();let w=(0..m*h).map(|i|value(i+2,0.25)).collect::<Vec<_>>();
        let s=Shape {op:Op::PoolScoresRows,heads:h as u32,dim:d as u32,taps:t as u32,rows:r as u32,first:first as u32,..rows(m)};
        let got=reference(s,[&q,&k,&w,&[],&[]],&[]).unwrap();
        for token in 0..m {let pools=(first+token)/t;
            let one=CausalShape {op:Causal::PoolScores,heads:h as u32,dim:d as u32,taps:0,rows:pools as u32,cols:0,epsilon:0.0,bound:0.0};
            let e=causal(one,[&q[token*h*d..(token+1)*h*d],&k[..pools*d],&w[token*h..(token+1)*h],&[]],&[]).unwrap();
            assert_eq!(bits(&e[..pools]),bits(&got[token*r..token*r+pools]),"token={token}");
        }
    }
    #[test]
    fn head_major_threads_visit_each_unit_once() {
        for (m,h) in [(1usize,1usize),(1,64),(5,64),(130,64),(7,3)] {let s=Shape {op:Op::ValuesRows,heads:h as u32,..rows(m)};
            let mut seen=vec![false;m*h];for thread in 0..m*h {let at=thread_unit(s,thread);assert!(!seen[at]);seen[at]=true;}
            assert_eq!(thread_unit(Shape {op:Op::SoftmaxRows,..s},m*h-1),m*h-1);}
    }
    #[test]
    fn permuted_rows_copy_each_source_row_bits() {
        let x=(0..12).map(|i|value(i,0.5)).collect::<Vec<_>>();let table=[2.0,0.0,1.0];
        let got=reference(Shape {op:Op::PermuteRows,rows:3,cols:4,first:3,..rows(1)},[&x,&table,&[],&[],&[]],&[]).unwrap();
        for row in 0..3 {assert_eq!(bits(&got[row*4..(row+1)*4]),bits(&x[table[row] as usize*4..(table[row] as usize+1)*4]));}
    }
    #[test]
    fn expert_place_returns_each_rank_row_and_token() {
        // Two experts, two ranks, two local rows. Units are [local row,tokens].
        let (d,parts)=(2usize,2usize);let tokens=[2usize,1];let slots=d*3;
        let mut big=vec![0.0f32;slots*parts];
        for rank in 0..parts {let mut at=rank*slots;for (e,&m) in tokens.iter().enumerate() {for row in 0..d {for t in 0..m {big[at]=(1000*e+100*(rank*d+row)+t) as f32;at+=1;}}}}
        let table=[0.0,2.0,1.0, 2.0,1.0,0.0, 0.0,2.0,0.0];
        let s=Shape {op:Op::ExpertPlace,rows:3,cols:4,dim:2,taps:2,slots:slots as u32,..rows(1)};
        let got=reference(s,[&big,&table,&[],&[],&[]],&[]).unwrap();
        for lane in 0..4 {assert_eq!(got[lane],(100*lane+1) as f32);assert_eq!(got[4+lane],(1000+100*lane) as f32);assert_eq!(got[8+lane],(100*lane) as f32);}
    }
    #[test]
    fn expert_place_tile_has_the_same_bytes_and_a_checked_table() {
        // Experts of sizes 3, 1, 4, 2 in expert order; rows come in token order, so groups interleave.
        let sizes=[3usize,1,4,2];let rows_n:usize=sizes.iter().sum();let (d,parts)=(4usize,2usize);let slots=d*rows_n;
        let mut owner=Vec::new();for (e,&n) in sizes.iter().enumerate() {for _ in 0..n {owner.push(e);}}
        // Row r takes the owner at position (r*7)%rows_n: a permutation, since 7 and 10 are coprime.
        let mut places=vec![0.0f32;3*rows_n];let mut starts=vec![0usize;sizes.len()];let mut sum=0;for (e,&n) in sizes.iter().enumerate() {starts[e]=sum;sum+=n;}
        let mut seen=vec![0usize;sizes.len()];
        for row in 0..rows_n {let e=owner[(row*7)%rows_n];places[3*row]=starts[e] as f32;places[3*row+1]=sizes[e] as f32;places[3*row+2]=seen[e] as f32;seen[e]+=1;}
        let big=(0..slots*parts).map(|i|value(i+5,0.0625)).collect::<Vec<_>>();
        let old=Shape {op:Op::ExpertPlace,rows:rows_n as u32,cols:(d*parts) as u32,dim:d as u32,taps:parts as u32,slots:slots as u32,..rows(1)};
        let tile=Shape {op:Op::ExpertPlaceTile,..old};
        let table=place_table(&places).unwrap();assert_eq!(table.len(),4*rows_n);assert_eq!(&table[..3*rows_n],&places[..]);
        for slot in 0..rows_n {let row=table[3*rows_n+slot] as usize;assert_eq!(places[3*row] as usize+places[3*row+2] as usize,slot);}
        let a=reference(old,[&big,&places,&[],&[],&[]],&[]).unwrap();let b=reference(tile,[&big,&table,&[],&[],&[]],&[]).unwrap();
        assert_eq!(bits(&a),bits(&b));
        // A table that is not a grouping keeps the original operation.
        let mut dup=places.clone();dup[2]=dup[5];dup[0]=dup[3];dup[1]=dup[4];assert!(place_table(&dup).is_none());
        let mut late=places.clone();late[2]=late[1];assert!(place_table(&late).is_none());
        let mut half=places.clone();half[0]=0.5;assert!(place_table(&half).is_none());
        let mut nan=places.clone();nan[1]=f32::NAN;assert!(place_table(&nan).is_none());
        assert!(place_table(&[]).is_none() && place_table(&[0.0,1.0]).is_none());
        let mut high=places.clone();high[0]=rows_n as f32;assert!(place_table(&high).is_none());
    }
    #[test]
    fn invalid_shapes_refuse() {
        assert!(Shape::new(Op::Transpose,0).is_none() || Shape {rows:1,cols:1,..Shape::new(Op::Transpose,0).unwrap()}.lengths().is_none());
        assert!(Shape {op:Op::ProjectRows,rows:u32::MAX,cols:u32::MAX,..rows(1)}.lengths().is_none());
        assert!(Shape {op:Op::StepRows,heads:1,dim:1025,..rows(1)}.lengths().is_none());
        assert!(Shape::wrapped(CausalShape {op:Causal::Pool,heads:0,dim:2,taps:2,rows:3,cols:0,epsilon:0.0,bound:0.0},2).unwrap().lengths().is_none());
    }
}
