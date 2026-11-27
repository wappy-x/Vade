//! The decode products of the looped text family split between the two Sparks of a cabled pair.
//!
//! Both Sparks hold every weight and run the same driver on the same chat. Each Spark keeps its own state, so the
//! attention, the mixes, the gates, the router and every small product run on both, with the same words. Only the
//! decode products whose weights are large are cut, and each cut keeps the arithmetic of one Spark:
//! - a dense product (the attention stack read by the query input, the output product, the head) is cut by its
//!   matrix rows. Each Spark runs the product of its own rows with the launch that one Spark runs for all of them; an
//!   output word is one chain over the columns of its row, so it is the same word on one Spark and on two. The cut
//!   falls on a 16-row band, where the device finds the same lossless panels (`Backend::panel_at`).
//! - the picked experts of one token row are cut by pick place: the lower half of the picks on the Spark of half
//!   0, the upper half on the Spark of half 1. Each Spark runs the gate, the up and the down products of its picks
//!   whole. The fold of the picks in pick order then runs on both Sparks over the same words.
//!
//! After a cut product the two Sparks trade their parts (`Pair::swap`), so both hold the whole result before any
//! launch reads it. A prompt piece runs whole on both Sparks: its words are those of one Spark by construction.
//! Every form of the weights that this file does not cut goes to the weights of one Spark unchanged, so the driver
//! takes the same launches and the same words as on one Spark.
//!
//! No line here names a model, a layer count or a size of a matrix: the cut follows the shapes of the weights.
use super::mix::{Message,Weight,Weights};
use super::operands::weights::{LoadedWeights,ResidentProduct};
use crate::{Result,bounded_decoder::Compute};
use std::{cell::{Cell,RefCell},ops::Range,time::Instant};

/// The rows of one band of the lossless panels: a cut at a multiple of it starts a sub-view that the device reads
/// with the panels of the whole matrix (`Backend::panel_at` refuses any other start).
const BAND:usize=16;

/// The rows or picks `0..count` of `half`: the lower part on half 0, the upper part on half 1. The lower part ends
/// at a whole band where `banded` (dense rows), at the middle pick otherwise.
pub fn part(count:usize,banded:bool,half:usize)->Range<usize> {
    let cut=if banded {(count/2).next_multiple_of(BAND).min(count)} else {count/2};
    [0..cut,cut..count][half].clone()
}
/// The whole result of a cut from the part of `half` (`mine`) and the other part (`peer`), each `parts` rows: row
/// after row, the lower part of the row first.
pub fn joined(mine:&[f32],peer:&[f32],parts:usize,half:usize)->Vec<f32> {
    let (own,theirs)=(mine.len()/parts,peer.len()/parts);
    let (lower,upper,low,high)=[(mine,peer,own,theirs),(peer,mine,theirs,own)][half];
    let mut whole=Vec::with_capacity(mine.len()+peer.len());
    for row in 0..parts {whole.extend_from_slice(&lower[row*low..(row+1)*low]);whole.extend_from_slice(&upper[row*high..(row+1)*high]);}
    whole
}
/// The half of each cut that a Spark of the pair computes: 0 the lower rows or picks, 1 the upper ones.
pub trait Half {fn half(&self)->usize;}
/// The wire between the two Sparks of a pair, for the values of the compute `C`.
pub trait Pair<C:Compute>:Half {
    /// The whole result of a cut from this Spark's part `local` (`parts` rows of the same count of words) and the
    /// other Spark's part (`parts` rows of `theirs` words): `[parts][own + theirs]`, in each row the lower part first.
    fn trade(&self,compute:&C,local:&C::Value,parts:usize,theirs:usize)->Result<C::Value>;
    /// Wait for the other Spark: a load of the weights takes a different time on each.
    fn ready(&self)->Result<()>;
    /// A fresh result of `words` words for a joined trade (`trade_into`). None: this wire has no joined trade, and
    /// the caller trades each product alone.
    /// A value of `words` words that no launch reads: the place of a cut product whose words this Spark keeps
    /// elsewhere (`Split::picked_products_strided_compact`).
    fn blank(&self,compute:&C,words:usize)->Result<C::Value> {compute.zeros(words)}
    /// `most` is the longest part of either Spark and `parts` the most products in one part.
    fn open(&self,_compute:&C,_words:usize,_most:usize,_parts:usize)->Result<Option<C::Value>> {Ok(None)}
    /// The joined trade of one cut whose part on this Spark is `locals` one after another (words `mine_at..` of the
    /// one row `whole`); the other Spark's part, `theirs` words, comes into words `peer_at..`.
    fn trade_into(&self,_compute:&C,_locals:&[&C::Value],_whole:&mut C::Value,_mine_at:usize,_peer_at:usize,_theirs:usize)->Result<()> {
        Err("this wire has no joined trade".into())
    }
    /// Contiguous `parts == 1` trade whose local half is the concatenation of `locals`. Default joins then trades.
    fn trade_parts(&self,compute:&C,locals:&[&C::Value],theirs:usize)->Result<C::Value> {
        let local=if locals.len()==1 {compute.range(locals[0],0,compute.words(locals[0]))?} else {compute.join(locals)?};
        self.trade(compute,&local,1,theirs)
    }
    /// Pack `[words]` BF16-valued FP32 words into `[words / 2]` packed BF16 words on device to halve RDMA wire
    /// volume on large prefill `parts == 1` trades. None: trade exact FP32 words as-is.
    fn pack_bf16(&self,_compute:&C,_local:&C::Value)->Result<Option<C::Value>> {Ok(None)}
    /// Unpack `[rows * dim / 2]` packed BF16 words back into `[rows * dim]` FP32 words on device (zero graph flush).
    fn unpack_bf16(&self,_compute:&C,_packed:&C::Value,_rows:usize,_dim:usize)->Result<Option<C::Value>> {Ok(None)}
    /// Arm 2-Spark expert-ID sharding (`0..experts/2` vs `experts/2..experts`) for the next `device_route_group`.
    fn arm_route_shard(&self,_compute:&C,_route:&C::Value,_rows:usize,_picks:usize,_experts:usize)->Result<bool> {Ok(false)}
    /// Consume `(mine_slots, peer_slots)` produced by an armed sharded `device_route_group` (clears armed state).
    fn take_route_shard(&self,_compute:&C)->Option<(Vec<usize>,Vec<usize>)> {None}
    /// Trade sharded BF16 expert down outputs and fold all `picks` per token in exact original pick order `0..picks`.
    #[allow(clippy::too_many_arguments)]
    fn trade_shard_fold(
        &self,_compute:&C,_local_bf16:&C::Value,_weights:&C::Value,_rows:usize,_picks:usize,_dim:usize,
        _mine_slots:&[usize],_peer_slots:&[usize],
    )->Result<Option<C::Value>> {Ok(None)}
}
/// The pair over the TCP and RDMA ring of `transport` (two members). One all-gather of two equal blocks a trade: each
/// block holds one Spark's part, padded to the longer part. The parts go through the host: a read, the gather and an
/// upload of the whole result.
pub struct RingPair {ring:RefCell<crate::transport::Ring>,half:usize}
impl RingPair {
    /// `ring` joins the two Sparks of the pair (a ring of two members): its member number is the half.
    pub fn new(ring:crate::transport::Ring)->Result<Self> {
        let half=ring.rank();
        if half>1 {return Err(format!("a split pair has halves 0 and 1; the ring gives member {half}"));}
        Ok(Self {ring:RefCell::new(ring),half})
    }
    /// Send this Spark's part and receive the other Spark's part, `theirs` words.
    fn swap(&self,mine:&[f32],theirs:usize)->Result<Vec<f32>> {
        let block=mine.len().max(theirs);
        let mut bytes=vec![0u8;2*block*4];
        for (word,out) in mine.iter().zip(bytes[self.half*block*4..].as_chunks_mut::<4>().0) {*out=word.to_le_bytes();}
        self.ring.borrow_mut().all_gather(&mut bytes,block*4)?;
        let peer=(1-self.half)*block*4;
        Ok(bytes[peer..peer+theirs*4].as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect())
    }
}
impl Half for RingPair {fn half(&self)->usize {self.half}}
impl<C:Compute> Pair<C> for RingPair {
    fn trade(&self,compute:&C,local:&C::Value,parts:usize,theirs:usize)->Result<C::Value> {
        let mine=compute.read(local)?;
        let peer=self.swap(&mine,parts*theirs)?;
        compute.upload(&joined(&mine,&peer,parts,self.half))
    }
    fn ready(&self)->Result<()> {self.ring.borrow_mut().barrier()}
}
/// The longest a take on the device waits for the other Spark's part. Both Sparks run the same steps, so a part
/// arrives within the time of one cut product on the other Spark; this bound only ends a run whose other Spark died.
#[cfg(target_os="linux")]
const TAKE_LIMIT_NS:u64=120_000_000_000;
/// Minimum half-payload words to split a `parts == 1` contiguous block trade across both RDMA rails (64 KiB).
#[cfg(target_os="linux")]
const DUAL_RAIL_CONTIG_WORDS:usize=16_384;
/// Minimum half-payload words to split a `parts > 1` strided trade across both RDMA rails (256 KiB): below this,
/// 2-launch `put_join + take` beats 5-launch dual-rail `scatter_copy + take_scatter`.
#[cfg(target_os="linux")]
const DUAL_RAIL_STRIDED_WORDS:usize=65_536;
/// Minimum FP32 words per half (1 MiB, i.e. prefill expert chunks) to BF16-pack `rn`-valued `parts == 1` expert
/// trades on device before RDMA transfer, halving wire bytes with zero graph flush and zero bit difference.
const BF16_TRADE_WORDS:usize=262_144;
/// The pair over the mapped mailbox of the device (`Backend::tp_mailbox`, moved by the wire thread of
/// `tensor_wire`): each trade is two launches on the ordered stream, with no host copy and no wait of the host
/// (`Device::pair_trade`). A second mailbox on the cable's second RDMA rail, when present, doubles bandwidth on
/// large block trades (`>= 64 KiB`). Multi-row messages larger than one slot chunk by rows over the mailbox before
/// any fallback to the host ring, preserving identical words without arbitrary caps.
#[cfg(target_os="linux")]
pub struct MailboxPair<'m,'d> {
    mailbox:&'m spark_gpu_probe::driver::TpMailbox<'d>,
    mailbox2:Option<&'m spark_gpu_probe::driver::TpMailbox<'d>>,
    ring:RingPair,
    seq:Cell<u64>,
    seq2:Cell<u64>,
}
#[cfg(target_os="linux")]
impl<'m,'d> MailboxPair<'m,'d> {
    /// `mailbox` is attached to the other Spark's mailbox by a running wire thread; `ring` joins the same two Sparks.
    pub fn new(mailbox:&'m spark_gpu_probe::driver::TpMailbox<'d>,ring:RingPair)->Self {Self::with_aux(mailbox,None,ring)}
    /// Primary mailbox plus an optional second mailbox on the cable's second RDMA rail.
    pub fn with_aux(mailbox:&'m spark_gpu_probe::driver::TpMailbox<'d>,mailbox2:Option<&'m spark_gpu_probe::driver::TpMailbox<'d>>,ring:RingPair)->Self {
        Self {mailbox,mailbox2,ring,seq:Cell::new(0),seq2:Cell::new(0)}
    }
}
#[cfg(target_os="linux")]
impl Half for MailboxPair<'_,'_> {fn half(&self)->usize {self.ring.half}}
#[cfg(target_os="linux")]
impl<'b,'d> Pair<crate::bounded_decoder::Device<'b,'d>> for MailboxPair<'_,'d> {
    fn trade(&self,compute:&crate::bounded_decoder::Device<'b,'d>,local:&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value,parts:usize,theirs:usize)->Result<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value> {
        let local_words=compute.words(local);
        let max_words=local_words.max(parts.saturating_mul(theirs));
        let dual_min=if parts==1 {DUAL_RAIL_CONTIG_WORDS} else {DUAL_RAIL_STRIDED_WORDS};
        if let Some(m2)=self.mailbox2 && max_words>=dual_min {
            let (s0,s1)=(self.seq.get()+1,self.seq2.get()+1);
            if let Some(whole)=compute.pair_trade_block_dual(self.mailbox,m2,local,parts,theirs,self.ring.half,s0,s1,TAKE_LIMIT_NS)? {
                self.seq.set(s0);self.seq2.set(s1);return Ok(whole);
            }
        }
        let seq=self.seq.get()+1;
        if let Some(whole)=compute.pair_trade(self.mailbox,local,parts,theirs,self.ring.half,seq,TAKE_LIMIT_NS)? {self.seq.set(seq);return Ok(whole);}
        // A long part (the head of a step of several rows, the summed experts of a prompt step): one put of it as it is.
        if let Some(whole)=compute.pair_trade_block(self.mailbox,local,parts,theirs,self.ring.half,seq,TAKE_LIMIT_NS)? {self.seq.set(seq);return Ok(whole);}
        // Multi-row payload larger than one mailbox slot: chunk by rows over the mailbox instead of host ring fallback.
        let slot=self.mailbox.slot_bytes()/4;
        if parts>1 && local_words.is_multiple_of(parts) {
            let own=local_words/parts;
            let row_max=own.max(theirs);
            if row_max>0 && row_max<=slot {
                let chunk=(slot/row_max).max(1);
                let mut outs=Vec::with_capacity(parts.div_ceil(chunk));
                for start in (0..parts).step_by(chunk) {
                    let n=chunk.min(parts-start);
                    let sub=compute.range(local,start*own,n*own)?;
                    outs.push(self.trade(compute,&sub,n,theirs)?);
                }
                return compute.join(&outs.iter().collect::<Vec<_>>());
            }
        }
        self.ring.trade(compute,local,parts,theirs)
    }
    fn ready(&self)->Result<()> {<RingPair as Pair<crate::bounded_decoder::Device<'b,'d>>>::ready(&self.ring)}
    fn blank(&self,compute:&crate::bounded_decoder::Device<'b,'d>,words:usize)->Result<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value> {compute.pair_blank(words)}
    fn open(&self,compute:&crate::bounded_decoder::Device<'b,'d>,words:usize,most:usize,parts:usize)->Result<Option<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value>> {
        compute.pair_open(self.mailbox,words,most,parts)
    }
    fn trade_into(&self,compute:&crate::bounded_decoder::Device<'b,'d>,locals:&[&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value],whole:&mut <crate::bounded_decoder::Device<'b,'d> as Compute>::Value,mine_at:usize,peer_at:usize,theirs:usize)->Result<()> {
        let seq=self.seq.get()+1;
        if !compute.pair_trade_into(self.mailbox,locals,whole,mine_at,peer_at,theirs,seq,TAKE_LIMIT_NS)? {return Err("pair trade: one joined put does not take the cut of a visit".into());}
        self.seq.set(seq);Ok(())
    }
    fn trade_parts(&self,compute:&crate::bounded_decoder::Device<'b,'d>,locals:&[&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value],theirs:usize)->Result<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value> {
        let own:usize=locals.iter().map(|v|compute.words(v)).sum();
        let m2=if own.max(theirs)>=DUAL_RAIL_CONTIG_WORDS {self.mailbox2} else {None};
        let (s0,s1)=(self.seq.get()+1,self.seq2.get()+1);
        if let Some((whole,used_dual))=compute.pair_trade_parts(self.mailbox,m2,locals,theirs,self.ring.half,s0,s1,TAKE_LIMIT_NS)? {
            self.seq.set(s0);if used_dual {self.seq2.set(s1);}return Ok(whole);
        }
        let local=compute.join(locals)?;
        self.trade(compute,&local,1,theirs)
    }
    fn pack_bf16(&self,compute:&crate::bounded_decoder::Device<'b,'d>,local:&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value)->Result<Option<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value>> {
        compute.pair_pack_bf16(local)
    }
    fn unpack_bf16(&self,compute:&crate::bounded_decoder::Device<'b,'d>,packed:&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value,rows:usize,dim:usize)->Result<Option<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value>> {
        compute.pair_unpack_bf16(packed,rows,dim)
    }
    fn arm_route_shard(&self,compute:&crate::bounded_decoder::Device<'b,'d>,route:&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value,rows:usize,picks:usize,experts:usize)->Result<bool> {
        compute.pair_arm_route_shard(route,rows,picks,experts,self.ring.half)
    }
    fn take_route_shard(&self,compute:&crate::bounded_decoder::Device<'b,'d>)->Option<(Vec<usize>,Vec<usize>)> {
        compute.pair_take_route_shard()
    }
    fn trade_shard_fold(
        &self,compute:&crate::bounded_decoder::Device<'b,'d>,
        local_bf16:&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value,
        weights:&<crate::bounded_decoder::Device<'b,'d> as Compute>::Value,
        rows:usize,picks:usize,dim:usize,mine_slots:&[usize],peer_slots:&[usize],
    )->Result<Option<<crate::bounded_decoder::Device<'b,'d> as Compute>::Value>> {
        compute.pair_trade_shard_fold(self.mailbox,self.mailbox2,local_bf16,weights,rows,picks,dim,mine_slots,peer_slots,&self.seq,&self.seq2,TAKE_LIMIT_NS)
    }
}
/// What the cut products of a run cost: the trades, their words (both ways) and the host time inside them (the wire
/// and the copies for the ring; the queueing of two launches for the mailbox).
#[derive(Debug,Clone,Copy,Default,PartialEq,Eq)]
pub struct Traded {pub swaps:u64,pub words:u64,pub ns:u64}

/// Causal-FLOP-balanced row split for prefill `window_rows_half`: returns `mid` aligned to `align` (`align >= 16`)
/// splitting `0..rows` into `0..mid` (half 0) and `mid..rows` (half 1) so both halves evaluate ~50% of causal key
/// interactions. None when `rows < 2 * align` or average live keys `<= 1024` (where RDMA trade exceeds compute saved).
pub fn window_split_mid(rows:usize,view:usize,first_position:u64,align:usize)->Option<usize> {
    if align==0 || rows<2*align || view<=1024 {return None;}
    let mut prefix=Vec::with_capacity(rows+1);
    prefix.push(0u64);
    let mut total=0u64;
    for r in 0..rows {
        total+=(first_position+r as u64+1).min(view as u64);
        prefix.push(total);
    }
    if total<=(rows as u64)*1024 {return None;}
    let target=total/2;
    let mut best=None;
    let mut best_diff=u64::MAX;
    for mid in (align..=rows-align).step_by(align) {
        let diff=prefix[mid].abs_diff(target);
        if diff<best_diff {best_diff=diff;best=Some(mid);}
    }
    best
}

/// Whole-matrix split index `k in 1..parts.len()` partitioning a stacked projection into `parts[..k]` (half 0) and
/// `parts[k..]` (half 1) at a `BAND`-aligned row boundary (`sum(parts[..k]) % BAND == 0`) with the closest 50/50 row
/// balance so both halves keep Vade-BF11-BP panels and trade in one contiguous `parts == 1` block with zero gather.
pub fn stack_split(parts:&[usize])->Option<usize> {
    if parts.len()<2 {return None;}
    let total:usize=parts.iter().sum();
    let mut left=0usize;
    let mut best:Option<(usize,usize)>=None;
    for (i,&size) in parts.iter().enumerate().take(parts.len()-1) {
        left=left.checked_add(size)?;
        let right=total-left;
        if left>0 && right>0 && left.is_multiple_of(BAND) {
            let diff=left.abs_diff(right);
            if best.is_none_or(|(_,d)|diff<d) {best=Some((i+1,diff));}
        }
    }
    best.map(|(k,_)|k)
}

/// The cut of a visit whose query, key and value inputs differ, between the call of the query stack and the call of
/// the value product: one trade for all three. The cut runs over the rows of the query stack, then those of the key,
/// then those of the value, as one matrix of `stack + key + value` rows, at a whole band inside the query stack.
/// So the Spark of half 0 runs its query rows, and the Spark of half 1 the other query rows and the key and the value
/// products whole, each with the launch of one Spark. The calls of the key and the value give views of `whole`, and
/// the call of the value queues the trade before it returns: no launch reads a word of `whole` before that.
struct Joint<X> {whole:X,own:Range<usize>,sides:[(Vec<Weight>,Vec<usize>,usize);2],next:usize,locals:Vec<X>}

/// The weights of one Spark of a pair: the weights of one Spark, and the cut of the large decode products.
/// `X` is the value of the compute (the joined cut of a visit, `Joint`, holds its parts between calls).
pub struct Split<'w,'l,V,P,X=()> {inner:&'w LoadedWeights<'l,V>,pair:&'w P,decode:Cell<bool>,bypass:Cell<bool>,traded:Cell<Traded>,joint:RefCell<Option<Joint<X>>>,
    /// The gated products of this Spark's rows of a step of several rows, between the gate and the down call.
    gated:RefCell<Option<Gated<X>>>,
    /// The gated products of this Spark's picks of one token row (`picked_gate`), between gate and down call.
    picked_one:RefCell<Option<PickedOne<X>>>}
/// Partition mode for multi-row small-batch decode experts (`M = 2..=16`, `picked_products_strided_compact`):
/// - `OneRow`: `M = 2` (1 row/half) runs `picked_gate` + `picked_products(Down)` so both halves use `expert_panels`.
/// - `Picks`: odd `M >= 3` splits by picks (`half_picks = picks / 2`) for exact 50/50 load balance on both Sparks.
/// - `Rows`: even `M >= 4` splits by whole rows (`M / 2` rows/half) with a single contiguous `parts == 1` trade.
enum GatedMode {OneRow(Range<usize>),Picks(usize),Rows(Range<usize>)}
struct Gated<X> {active:X,route:X,mode:GatedMode}
/// The gated products of this Spark's picks of one token row (`picked_gate`), kept between `picked_gate` and
/// `picked_products(Down)` so no zero-pad or join launch runs on C1.
struct PickedOne<X> {active:X,own:Range<usize>,picks:usize}
/// Expert-ID shard metadata (`0..experts/2` on half 0, `experts/2..experts` on half 1) for a grouped route plan:
/// `mine_slots` and `peer_slots` partition `0..rows*picks` in each half's exact grouped expert order.
pub struct ShardInfo {pub rows:usize,pub picks:usize,pub mine_slots:Vec<usize>,pub peer_slots:Vec<usize>}
/// The route of a step of several rows: the plan of all rows (or this Spark's expert shard when `shard` is `Some`),
/// optional row-cut sub-plan (`own`), and optional expert-ID shard metadata (`shard`).
pub struct SplitPlan<R,X> {full:R,own:Option<(R,X,Range<usize>,usize)>,shard:Option<ShardInfo>}
impl<'w,'l,V,P:Half,X> Split<'w,'l,V,P,X> {
    pub fn new(inner:&'w LoadedWeights<'l,V>,pair:&'w P)->Self {Self {inner,pair,decode:Cell::new(true),bypass:Cell::new(false),traded:Cell::new(Traded::default()),joint:RefCell::new(None),gated:RefCell::new(None),picked_one:RefCell::new(None)}}
    /// Cell toggled true on Rank 0 during local-only drafter execution so no weight call ever initiates a pair trade.
    pub fn bypass(&self)->&Cell<bool> {&self.bypass}
    /// Every call but the key and the value product of an open joined cut finds none open.
    fn idle(&self,call:&str)->Result<()> {if self.joint.borrow().is_some() || self.gated.borrow().is_some() || self.picked_one.borrow().is_some() {Err(format!("tp2: a call ({call}) came between the query and the value products of a joined cut"))} else {Ok(())}}
    /// The trades so far.
    pub fn traded(&self)->Traded {self.traded.get()}
    /// The rows or picks `0..count` of this Spark: the lower part on half 0, the upper part on half 1. The lower part
    /// ends at a whole band where `banded` (dense rows), at the middle pick otherwise.
    fn mine(&self,count:usize,banded:bool)->Range<usize> {part(count,banded,self.pair.half())}
    /// Trade this Spark's part `local` (`parts` rows) for the other part (`parts` rows of `theirs` words): the whole
    /// result, row after row, the lower part of each row first.
    fn whole<C:Compute>(&self,compute:&C,local:&C::Value,parts:usize,theirs:usize)->Result<C::Value> where P:Pair<C> {
        let began=Instant::now();
        let whole=self.pair.trade(compute,local,parts,theirs)?;
        let t=self.traded.get();
        self.traded.set(Traded {swaps:t.swaps+1,words:t.words+compute.words(&whole) as u64,ns:t.ns+began.elapsed().as_nanos() as u64});
        Ok(whole)
    }
    /// Contiguous `parts == 1` trade whose local half is `locals` concatenated in order.
    fn whole_parts<C:Compute>(&self,compute:&C,locals:&[&C::Value],theirs:usize)->Result<C::Value> where P:Pair<C> {
        let began=Instant::now();
        let whole=self.pair.trade_parts(compute,locals,theirs)?;
        let t=self.traded.get();
        self.traded.set(Traded {swaps:t.swaps+1,words:t.words+compute.words(&whole) as u64,ns:t.ns+began.elapsed().as_nanos() as u64});
        Ok(whole)
    }
    /// Lossless zero-flush BF16-packed `parts == 1` trade for large prefill `rn(sum)` expert sums
    /// (`device_routed_experts*`): packs via `PREFILL_OP_PACK`, trades half the bytes on dual-rail RDMA, and unpacks
    /// via resident `Kind::FastStreamOps` (`fast_route_fold_half_into`, `picks = 1`). Symmetric across ranks.
    fn whole_bf16<C:Compute>(&self,compute:&C,local:&C::Value,rows:usize,dim:usize,theirs:usize)->Result<C::Value> where P:Pair<C> {
        use spark_gpu_probe::fast_ops;
        let own=compute.words(local);
        if !self.decode.get() && own.min(theirs)>=BF16_TRADE_WORDS && own.is_multiple_of(8) && theirs.is_multiple_of(8)
            && dim>0 && dim.is_multiple_of(2) && own+theirs==rows.saturating_mul(dim)
            && fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD,rows,dim,1,fast_ops::STREAM_OPS_THREADS)
            && let Some(packed)=self.pair.pack_bf16(compute,local)?
        {
            let traded=self.whole(compute,&packed,1,theirs/2)?;
            return self.pair.unpack_bf16(compute,&traded,rows,dim)?.ok_or_else(||"tp2: bf16 wire unpack unavailable after bf16 pack".into());
        }
        self.whole(compute,local,1,theirs)
    }
    /// A dense product cut by matrix rows: this Spark's rows of `value[words]` (`rows x columns` BF16 values) on
    /// `inputs` input rows, the launch of one Spark on those rows, then the trade. The result is `[inputs][rows]`.
    #[allow(clippy::too_many_arguments)]
    fn rows_cut<C:ResidentProduct<V>>(&self,compute:&C,value:&V,words:Range<usize>,rows:usize,columns:usize,x:&C::Value,inputs:usize)->Result<C::Value> where P:Pair<C> {
        let own=self.mine(rows,true);
        let row_words=columns/2;
        let local=compute.resident_product_rows(value,words.start+own.start*row_words..words.start+own.end*row_words,own.len(),columns,x,inputs,true)?;
        self.whole(compute,&local,inputs,rows-own.len())
    }
    /// The picks of this Spark among `ids` (picked numbers of one token row), as a range of the value.
    fn own_ids<C:ResidentProduct<V>>(&self,compute:&C,ids:&C::Value)->Result<(Range<usize>,usize,C::Value)> {
        let picks=compute.words(ids);
        let own=self.mine(picks,false);
        let part=compute.range(ids,own.start,own.len())?;
        Ok((own,picks,part))
    }
    /// `local` (this Spark's picks, `own.len()` rows) inside zero rows for the other picks: the shape of the products
    /// of all picks, which the driver checks, with the words of the other picks never read (`picked_products` of the
    /// down projection reads this Spark's rows only).
    fn padded<C:ResidentProduct<V>>(&self,compute:&C,local:C::Value,own:&Range<usize>,picks:usize)->Result<C::Value> {
        let each=compute.words(&local)/own.len();
        let mut parts=Vec::with_capacity(3);
        if own.start>0 {parts.push(compute.zeros(own.start*each)?);}
        parts.push(local);
        if own.end<picks {parts.push(compute.zeros((picks-own.end)*each)?);}
        compute.join(&parts.iter().collect::<Vec<_>>())
    }
    /// The stack that the query input reads first, of a decode step of `rows` rows (two or more) of `x`, where both
    /// Sparks get rows of it: its value, words, the rows of each matrix and the columns.
    #[allow(clippy::type_complexity)]
    fn batch_query_stack<C:ResidentProduct<V>>(&self,compute:&C,weights:&[Weight],x:&C::Value,rows:usize)->Result<Option<(&V,Range<usize>,Vec<usize>,usize)>> {
        if !self.decode.get() || rows<2 || !Self::query_stack(weights) {return Ok(None);}
        let Some((value,words,parts,columns))=self.inner.run(weights)? else {return Ok(None)};
        Ok((Self::cuts(parts.iter().sum(),true) && compute.words(x)==rows*columns).then_some((value,words,parts,columns)))
    }
    /// True for the stack that the query input reads first: the cut covers all its rows.
    fn query_stack(weights:&[Weight])->bool {matches!(weights.first(),Some(Weight::Query(_)))}
    /// True where both Sparks get rows of a cut of `rows` (or picks, when not `banded`).
    fn cuts(count:usize,banded:bool)->bool {!part(count,banded,0).is_empty() && !part(count,banded,1).is_empty()}
}
impl<V,P:Half,X> Split<'_,'_,V,P,X> {
    /// Expert-ID sharded MoE (`0..experts/2` on half 0, `experts/2..experts` on half 1): runs `GateUp` and `Down`
    /// only on this half's experts into packed BF16, trades per-pick BF16 outputs over dual-rail RDMA, scatters into
    /// exact pick order `0..picks`, and folds bit-identically to 1-Spark.
    #[allow(clippy::too_many_arguments)]
    fn shard_cut<C:ResidentProduct<V>>(&self,compute:&C,layer:usize,route:&SplitPlan<C::RoutePlan,C::Value>,x:&C::Value,caps:[f32;4],weights:&C::Value)->Result<Option<C::Value>> where P:Pair<C> {
        let Some(shard)=route.shard.as_ref() else {return Ok(None)};
        let own_picks=shard.mine_slots.len();
        if own_picks==0 {return Ok(None);}
        let Some(active)=self.inner.device_routed_gate(compute,layer,&route.full,x,caps)? else {return Ok(None)};
        let Some(downs_raw)=self.inner.device_routed_products(compute,Weight::Down(layer,0),&route.full,&active)? else {return Ok(None)};
        let raw_words=compute.words(&downs_raw);
        if !raw_words.is_multiple_of(own_picks) {return Err("tp2: sharded down output differs from own picks".into());}
        let d_rows=raw_words/own_picks;
        if !d_rows.is_multiple_of(2) {return Err("tp2: sharded down width must be even".into());}
        let local_bf16=compute.range(&downs_raw,0,own_picks*(d_rows/2))?;
        let began=Instant::now();
        let Some(sum)=self.pair.trade_shard_fold(compute,&local_bf16,weights,shard.rows,shard.picks,d_rows,&shard.mine_slots,&shard.peer_slots)? else {
            return Err("tp2: sharded expert trade unavailable after sharded gemm".into());
        };
        let t=self.traded.get();
        let traded_words=(shard.rows*shard.picks*(d_rows/2)) as u64;
        self.traded.set(Traded {swaps:t.swaps+1,words:t.words+traded_words,ns:t.ns+began.elapsed().as_nanos() as u64});
        Ok(Some(sum))
    }
    /// The summed experts of a step of several rows, cut by rows where `route` holds this Spark's part: `own` runs
    /// on this Spark's rows of `x` (whole rows of words), then the two Sparks trade the sums. `full` otherwise, and
    /// where `own` has no form for this Spark's rows: both Sparks decide that alike, from the same rows.
    fn row_cut<C,R>(&self,compute:&C,route:&SplitPlan<R,X>,x:&X,own:impl FnOnce(&R,&X,&X)->Result<Option<X>>,full:impl FnOnce(&R)->Result<Option<X>>)->Result<Option<X>> where C:Compute<Value=X>,P:Pair<C> {
        let Some((own_plan,own_weights,rows,all))=route.own.as_ref() else {return full(&route.full)};
        if !compute.words(x).is_multiple_of(*all) {return full(&route.full);}
        let each=compute.words(x)/all;
        let own_x=compute.range(x,rows.start*each,rows.len()*each)?;
        // No form of the call for this Spark's rows (a form that takes only steps of more rows): the call of all rows.
        let Some(local)=own(own_plan,&own_x,own_weights)? else {return full(&route.full)};
        let width=compute.words(&local)/rows.len();
        self.whole_bf16(compute,&local,*all,width,(all-rows.len())*width).map(Some)
    }
    /// The rows of the key (`value` false) or the value product of `layer` with its tap product, where the layer
    /// has one (the driver then asks for both in one stack call), and their columns.
    fn side(&self,layer:usize,value:bool)->Result<Option<(Vec<Weight>,Vec<usize>,usize)>> {
        let (main,taps)=if value {(Weight::Value(layer),Weight::ValueTapProjection(layer))} else {(Weight::Key(layer),Weight::KeyTapProjection(layer))};
        let Some((_,_,rows,columns))=self.inner.matrix(main)? else {return Ok(None)};
        Ok(Some(match self.inner.matrix(taps).ok().flatten() {
            Some((_,_,tap_rows,tap_columns)) if tap_columns==columns=>(vec![main,taps],vec![rows,tap_rows],columns),
            Some(_)=>return Ok(None),
            None=>(vec![main],vec![rows],columns),
        }))
    }
    /// Open the joined cut of a visit at the call of its query stack (`Joint`), where the stack is that of the query
    /// input alone and the key and the value products of the layer have the same columns. None: the cut does not
    /// fall inside the query stack, or the wire has no joined trade of this shape; the caller cuts as before.
    fn joint_open<C>(&self,compute:&C,weights:&[Weight],x:&X)->Result<Option<Vec<X>>> where C:ResidentProduct<V>+Compute<Value=X>,P:Pair<C> {
        let &[Weight::Query(layer),Weight::Relative(l1),Weight::Sink(l2),Weight::OutputGate(l3)]=weights else {return Ok(None)};
        if !self.decode.get() || [l1,l2,l3]!=[layer;3] {return Ok(None);}
        let Some((value,words,parts,columns))=self.inner.run(weights)? else {return Ok(None)};
        let (Some((key_names,key_rows,key_columns)),Some((value_names,value_rows,value_columns)))=(self.side(layer,false)?,self.side(layer,true)?) else {return Ok(None)};
        let stack=parts.iter().sum::<usize>();let (key,val)=(key_rows.iter().sum::<usize>(),value_rows.iter().sum::<usize>());let total=stack+key+val;
        if key_columns!=columns || value_columns!=columns || compute.words(x)!=columns {return Ok(None);}
        let cut=part(total,true,0).end;
        if cut==0 || cut>=stack {return Ok(None);}
        let Some(whole)=self.pair.open(compute,total,cut.max(total-cut),1+key_names.len()+value_names.len())? else {return Ok(None)};
        let own=self.mine(total,true);
        let mut locals=Vec::with_capacity(5);
        let rows=own.start..own.end.min(stack);let row_words=columns/2;
        if !rows.is_empty() {locals.push(compute.resident_product_rows(value,words.start+rows.start*row_words..words.start+rows.end*row_words,rows.len(),columns,x,1,true)?);}
        let mut start=0;
        let views=parts.iter().map(|&n|{let view=compute.range(&whole,start,n);start+=n;view}).collect::<Result<Vec<_>>>()?;
        *self.joint.borrow_mut()=Some(Joint {whole,own,sides:[(key_names,key_rows,stack),(value_names,value_rows,stack+key)],next:0,locals});
        Ok(Some(views))
    }
    /// The key, then the value call of an open joined cut (one product, or a stack call of the product and its tap
    /// product): this Spark's part (the whole products on half 1, none on half 0), and views of their words in the
    /// whole result. The value call queues the one trade of the visit.
    fn joint_call<C>(&self,compute:&C,weights:&[Weight],x:&X)->Result<Vec<X>> where C:ResidentProduct<V>+Compute<Value=X>,P:Pair<C> {
        let mut slot=self.joint.borrow_mut();
        let joint=slot.as_mut().ok_or("tp2: no joined cut is open")?;
        let (names,rows,at)=joint.sides.get(joint.next).ok_or("tp2: a joined cut has no call left")?;
        if names.as_slice()!=weights {return Err(format!("tp2: a call of {weights:?} came between the query and the value products of a joined cut (next {names:?})"));}
        let (rows,at)=(rows.clone(),*at);
        // The cut falls inside the query stack: the part of half 1 holds the key and the value whole.
        if joint.own.end>at {
            let made=if weights.len()==1 {vec![self.inner.product(compute,weights[0],x)?]} else {self.inner.stack_products(compute,weights,x)?};
            joint.locals.extend(made);
        }
        let mut start=at;
        let views=rows.iter().map(|&n|{let view=compute.range(&joint.whole,start,n);start+=n;view}).collect::<Result<Vec<_>>>()?;
        joint.next+=1;
        if joint.next<joint.sides.len() {return Ok(views);}
        let Joint {mut whole,own,locals,..}=slot.take().ok_or("tp2: no joined cut is open")?;
        let total=compute.words(&whole);
        let other=part(total,true,1-self.pair.half());
        let began=Instant::now();
        self.pair.trade_into(compute,&locals.iter().collect::<Vec<_>>(),&mut whole,own.start,other.start,other.len())?;
        let t=self.traded.get();
        self.traded.set(Traded {swaps:t.swaps+1,words:t.words+total as u64,ns:t.ns+began.elapsed().as_nanos() as u64});
        Ok(views)
    }
}
impl<C:ResidentProduct<V>,V,P:Pair<C>> Weights<C> for Split<'_,'_,V,P,C::Value> {
    type RoutePlan=SplitPlan<C::RoutePlan,C::Value>;
    fn arithmetic_phase(&self,decode:bool)->bool {self.decode.set(decode);<LoadedWeights<'_,V> as Weights<C>>::arithmetic_phase(self.inner,decode)}
    fn words(&self,weight:Weight)->Result<&[f32]> {<LoadedWeights<'_,V> as Weights<C>>::words(self.inner,weight)}
    fn product(&self,compute:&C,weight:Weight,x:&C::Value)->Result<C::Value> {
        if self.bypass.get() {return self.inner.product(compute,weight,x);}
        if self.joint.borrow().is_some() {return self.joint_call(compute,&[weight],x)?.pop().ok_or_else(||"tp2: a joined product is missing".to_owned());}
        if self.decode.get() && matches!(weight,Weight::Output(_)|Weight::Head) && let Some((value,words,rows,columns))=self.inner.matrix(weight)? && Self::cuts(rows,true) {
            return self.rows_cut(compute,value,words,rows,columns,x,1);
        }
        self.inner.product(compute,weight,x)
    }
    fn stored_rows(&self,compute:&C,weight:Weight,tokens:&[u32])->Result<Option<C::Value>> {self.idle("stored_rows")?;self.inner.stored_rows(compute,weight,tokens)}
    fn stack_products(&self,compute:&C,weights:&[Weight],x:&C::Value)->Result<Vec<C::Value>> {
        if self.bypass.get() {return self.inner.stack_products(compute,weights,x);}
        if self.joint.borrow().is_some() {return self.joint_call(compute,weights,x);}
        if let Some(parts)=self.joint_open(compute,weights,x)? {return Ok(parts);}
        if self.decode.get() && Self::query_stack(weights) && let Some((value,words,parts,columns))=self.inner.run(weights)? && Self::cuts(parts.iter().sum(),true) {
            let y=self.rows_cut(compute,value,words,parts.iter().sum(),columns,x,1)?;
            let mut start=0;
            return parts.iter().map(|&n|{let part=compute.range(&y,start,n);start+=n;part}).collect();
        }
        self.inner.stack_products(compute,weights,x)
    }
    fn product_rows(&self,compute:&C,weight:Weight,x:&C::Value,rows:usize)->Result<C::Value> {
        if self.bypass.get() {return self.inner.product_rows(compute,weight,x,rows);}
        self.idle("product_rows")?;
        let cut_weight=matches!(weight,Weight::Head) || (matches!(weight,Weight::Output(_)) && rows<=spark_gpu_probe::few_product::FEW_MAX);
        if self.decode.get() && rows>0 && compute.words(x).is_multiple_of(rows) && cut_weight
            && let Some((value,words,matrix_rows,columns))=self.inner.matrix(weight)? && Self::cuts(matrix_rows,true) {
            return self.rows_cut(compute,value,words,matrix_rows,columns,x,rows);
        }
        self.inner.product_rows(compute,weight,x,rows)
    }
    fn products(&self,compute:&C,weights:&[Weight],x:&C::Value,shared:bool)->Result<Option<C::Value>> {self.idle("products")?;self.inner.products(compute,weights,x,shared)}
    fn group_products(&self,compute:&C,weights:&[Weight],xs:&[&C::Value])->Result<Option<C::Value>> {self.idle("group_products")?;self.inner.group_products(compute,weights,xs)}
    fn stack_product_rows(&self,compute:&C,weights:&[Weight],x:&C::Value,rows:usize)->Result<Option<C::Value>> {
        if self.bypass.get() {return self.inner.stack_product_rows(compute,weights,x,rows);}
        self.idle("stack_product_rows")?;
        // The query stack of a decode step of several rows: cut by matrix rows, as for one row; `[rows][all rows of the stack]`.
        if let Some((value,words,parts,columns))=self.batch_query_stack(compute,weights,x,rows)? {
            return self.rows_cut(compute,value,words,parts.iter().sum(),columns,x,rows).map(Some);
        }
        self.inner.stack_product_rows(compute,weights,x,rows)
    }
    fn picked_products(&self,compute:&C,first:Weight,ids:&C::Value,x:&C::Value,shared:bool,token_rows:usize)->Result<Option<C::Value>> {
        if self.bypass.get() {return self.inner.picked_products(compute,first,ids,x,shared,token_rows);}
        if let Some(PickedOne {active,own,picks})=self.picked_one.borrow_mut().take() {
            if !matches!(first,Weight::Down(..)) || shared || token_rows!=1 || compute.words(ids)!=picks {
                return Err("tp2: the gated products of one row met no down call".into());
            }
            let part=compute.range(ids,own.start,own.len())?;
            let Some(local)=self.inner.picked_products(compute,first,&part,&active,false,1)? else {return Ok(None)};
            let width=compute.words(&local)/own.len();
            return self.whole(compute,&local,1,(picks-own.len())*width).map(Some);
        }
        self.idle("picked_products")?;
        if !self.decode.get() || token_rows!=1 || !Self::cuts(compute.words(ids),false) {return self.inner.picked_products(compute,first,ids,x,shared,token_rows);}
        let (own,picks,part)=self.own_ids(compute,ids)?;
        match first {
            // The down products read one input row a pick: this Spark's rows of the gated products.
            Weight::Down(..) if !shared && picks>0 && compute.words(x).is_multiple_of(picks) => {
                let each=compute.words(x)/picks;
                let input=compute.range(x,own.start*each,own.len()*each)?;
                let Some(local)=self.inner.picked_products(compute,first,&part,&input,false,1)? else {return Ok(None)};
                let width=compute.words(&local)/own.len();
                self.whole(compute,&local,1,(picks-own.len())*width).map(Some)
            },
            // The gate and up products of one shared row (the form without the fused gate): this Spark's picks only.
            Weight::GateUp(..) if shared => {
                let Some(local)=self.inner.picked_products(compute,first,&part,x,true,1)? else {return Ok(None)};
                self.padded(compute,local,&own,picks).map(Some)
            },
            _ => self.inner.picked_products(compute,first,ids,x,shared,token_rows),
        }
    }
    fn picked_gate(&self,compute:&C,layer:usize,ids:&C::Value,x:&C::Value,shared:bool,caps:[f32;4])->Result<Option<C::Value>> {
        if self.bypass.get() {return self.inner.picked_gate(compute,layer,ids,x,shared,caps);}
        self.idle("picked_gate")?;
        if !self.decode.get() || !shared || !Self::cuts(compute.words(ids),false) {return self.inner.picked_gate(compute,layer,ids,x,shared,caps);}
        let (own,picks,part)=self.own_ids(compute,ids)?;
        let Some(local)=self.inner.picked_gate(compute,layer,&part,x,true,caps)? else {return Ok(None)};
        // Avoid 144 wasted zero-pad/join launches per token: store this Spark's active rows for `picked_products(Down)`.
        let each=compute.words(&local)/own.len();
        let blank=self.pair.blank(compute,picks*each)?;
        *self.picked_one.borrow_mut()=Some(PickedOne {active:local,own,picks});
        Ok(Some(blank))
    }
    fn picked_products_strided(&self,compute:&C,first:Weight,route:&C::Value,rows:usize,picks:usize,id_stride:usize,x:&C::Value,repeat:usize,caps:Option<[f32;4]>)->Result<Option<C::Value>> {
        self.idle("picked_products_strided")?;self.inner.picked_products_strided(compute,first,route,rows,picks,id_stride,x,repeat,caps)
    }
    fn picked_products_strided_compact(&self,compute:&C,first:Weight,route:&C::Value,rows:usize,picks:usize,id_stride:usize,x:&C::Value,repeat:usize,caps:Option<[f32;4]>,compact:Option<&C::Value>)->Result<Option<C::Value>> {
        if self.bypass.get() {return self.inner.picked_products_strided_compact(compute,first,route,rows,picks,id_stride,x,repeat,caps,compact);}
        if let Some(Gated {active,route:own_route,mode})=self.gated.borrow_mut().take() {
            if !matches!(first,Weight::Down(..)) || repeat!=1 {return Err("tp2: the gated products of a row cut met no down call".into());}
            match mode {
                GatedMode::OneRow(own) => {
                    let Some(local)=self.inner.picked_products(compute,first,&own_route,&active,false,1)? else {
                        return Err("tp2: the single-row down products of a 2-row cut have no call".into());
                    };
                    let width=compute.words(&local)/picks.max(1);
                    return self.whole(compute,&local,1,(rows-own.len())*picks*width).map(Some);
                }
                GatedMode::Picks(half_picks) => {
                    let Some(local)=self.inner.picked_products_strided_compact(compute,first,&own_route,rows,half_picks,id_stride,&active,1,None,None)? else {
                        return Err("tp2: the down products of a pick cut have no call".into());
                    };
                    let width=compute.words(&local)/(rows*half_picks).max(1);
                    return self.whole(compute,&local,rows,(picks-half_picks)*width).map(Some);
                }
                GatedMode::Rows(own) => {
                    let Some(local)=self.inner.picked_products_strided_compact(compute,first,&own_route,own.len(),picks,id_stride,&active,1,None,None)? else {
                        return Err("tp2: the down products of a row cut have no call".into());
                    };
                    let width=compute.words(&local)/(own.len()*picks).max(1);
                    return self.whole(compute,&local,1,(rows-own.len())*picks*width).map(Some);
                }
            }
        }
        self.idle("picked_products_strided_compact")?;
        if let (Weight::GateUp(layer,0),Some(cap_words))=(first,caps) && repeat==picks && rows>1 && Self::cuts(rows,false)
            && id_stride>=picks && compute.words(x).is_multiple_of(rows) && compute.words(route)==rows*id_stride
        {
            let width=compute.words(x)/rows;
            // `M = 2`: each half owns 1 token row -> use `picked_gate` + `picked_products(Down)` for `expert_panels`.
            if rows==2 {
                let own=self.mine(rows,false);
                let own_ids=compute.range(route,own.start*id_stride,picks)?;
                let own_x=compute.range(x,own.start*width,width)?;
                if let Some(active)=self.inner.picked_gate(compute,layer,&own_ids,&own_x,true,cap_words)? {
                    let inner=compute.words(&active)/picks;
                    let blank=self.pair.blank(compute,rows*picks*inner)?;
                    *self.gated.borrow_mut()=Some(Gated {active,route:own_ids,mode:GatedMode::OneRow(own)});
                    return Ok(Some(blank));
                }
            }
            // Odd `M >= 3` (e.g. `M = 3, 5, 7` draft verify): split by picks (`picks / 2` per half) for 50/50 balance.
            if rows%2==1 && picks>=2 && picks.is_multiple_of(2) && id_stride>picks/2 {
                let half_picks=picks/2;
                let total_words=rows*id_stride;
                let own_route=if self.pair.half()==0 {
                    compute.range(route,0,total_words)?
                } else {
                    let tail=compute.range(route,half_picks,total_words-half_picks)?;
                    let head=compute.range(route,0,half_picks)?;
                    compute.join(&[&tail,&head])?
                };
                if let Some(active)=self.inner.picked_products_strided_compact(compute,first,&own_route,rows,half_picks,id_stride,x,half_picks,caps,None)? {
                    let inner=compute.words(&active)/(rows*half_picks);
                    let blank=self.pair.blank(compute,rows*picks*inner)?;
                    *self.gated.borrow_mut()=Some(Gated {active,route:own_route,mode:GatedMode::Picks(half_picks)});
                    return Ok(Some(blank));
                }
            }
            // Even `M >= 4` (or fallback): split by whole rows (`M / 2` rows/half) with contiguous `parts == 1` trade.
            let own=self.mine(rows,false);
            let own_route=compute.range(route,own.start*id_stride,own.len()*id_stride)?;
            let own_x=compute.range(x,own.start*width,own.len()*width)?;
            let Some(active)=self.inner.picked_products_strided_compact(compute,first,&own_route,own.len(),picks,id_stride,&own_x,repeat,caps,None)? else {return Ok(None)};
            let inner=compute.words(&active)/(own.len()*picks);
            let blank=self.pair.blank(compute,rows*picks*inner)?;
            *self.gated.borrow_mut()=Some(Gated {active,route:own_route,mode:GatedMode::Rows(own)});
            return Ok(Some(blank));
        }
        self.inner.picked_products_strided_compact(compute,first,route,rows,picks,id_stride,x,repeat,caps,compact)
    }
    fn stack_products_rows(&self,compute:&C,weights:&[Weight],x:&C::Value,rows:usize)->Result<Option<Vec<C::Value>>> {
        if self.bypass.get() {return self.inner.stack_products_rows(compute,weights,x,rows);}
        self.idle("stack_products_rows")?;
        // Split the query stack at a band-aligned whole-matrix boundary (`stack_split`) so both halves keep lossless
        // Vade-BF11-BP panels and write directly into per-weight buffers, eliminating 48x `prompt_rows` gathers.
        // When a shared QKV stack has trailing K/V (`weights.len() >= k + 3`), pair `weights[tail]` with half 0 and
        // `weights[tail + 1]` with half 1 to balance `4480` vs `4120` rows instead of `3456` vs `5144`.
        if let Some((_,_,parts,columns))=self.batch_query_stack(compute,weights,x,rows)? {
            if let Some(k)=stack_split(&parts)
                && k<=spark_gpu_probe::few_product::FEW_SPLIT_MAX
                && parts.iter().all(|&w|w.is_multiple_of(2))
            {
                let left:usize=parts[..k].iter().sum();
                let right:usize=parts[k..].iter().sum();
                let tail=weights.len().saturating_sub(2);
                if weights.len()>=k+3 && tail-k<=spark_gpu_probe::few_product::FEW_SPLIT_MAX {
                    let mid_rows:usize=parts[k..tail].iter().sum();
                    let (pa,pb)=(parts[tail],parts[tail+1]);
                    if (left+pa).abs_diff(mid_rows+pb)<left.abs_diff(right)
                        && spark_gpu_probe::tc_rows::tc_shape_admitted(left,columns,1)
                        && spark_gpu_probe::tc_rows::tc_shape_admitted(mid_rows,columns,1)
                        && self.inner.run(&weights[..k])?.is_some()
                        && self.inner.run(&weights[k..tail])?.is_some()
                    {
                        let (mine_w,extra_w,other_rows)=if self.pair.half()==0 {(&weights[..k],weights[tail],mid_rows+pb)} else {(&weights[k..tail],weights[tail+1],left+pa)};
                        if let Some(mut mine)=self.inner.stack_products_rows(compute,mine_w,x,rows)? {
                            mine.push(self.inner.product_rows(compute,extra_w,x,rows)?);
                            let refs:Vec<&C::Value>=mine.iter().collect();
                            let whole=self.whole_parts(compute,&refs,rows*other_rows)?;
                            drop(mine);
                            let mut out=Vec::with_capacity(parts.len());
                            let mut at=0;
                            for &size in &parts[..k] {out.push(compute.range(&whole,at,rows*size)?);at+=rows*size;}
                            let part_a=compute.range(&whole,at,rows*pa)?;at+=rows*pa;
                            for &size in &parts[k..tail] {out.push(compute.range(&whole,at,rows*size)?);at+=rows*size;}
                            let part_b=compute.range(&whole,at,rows*pb)?;
                            out.push(part_a);out.push(part_b);
                            return Ok(Some(out));
                        }
                    }
                }
                if weights.len()-k<=spark_gpu_probe::few_product::FEW_SPLIT_MAX
                    && spark_gpu_probe::tc_rows::tc_shape_admitted(left,columns,1)
                    && spark_gpu_probe::tc_rows::tc_shape_admitted(right,columns,1)
                    && self.inner.run(&weights[..k])?.is_some()
                    && self.inner.run(&weights[k..])?.is_some()
                {
                    let (mine_w,other_rows)=if self.pair.half()==0 {(&weights[..k],right)} else {(&weights[k..],left)};
                    if let Some(mine)=self.inner.stack_products_rows(compute,mine_w,x,rows)? {
                        let refs:Vec<&C::Value>=mine.iter().collect();
                        let whole=self.whole_parts(compute,&refs,rows*other_rows)?;
                        drop(mine);
                        let mut start=0;
                        return parts.iter().map(|&size|{let p=compute.range(&whole,start,rows*size);start+=rows*size;p}).collect::<Result<Vec<_>>>().map(Some);
                    }
                }
            }
            return Ok(None);
        }
        self.inner.stack_products_rows(compute,weights,x,rows)
    }
    fn product_rows_pair(&self,compute:&C,weight:Weight,x1:&C::Value,x2:&C::Value,rows:usize)->Result<Option<C::Value>> {self.idle("product_rows_pair")?;self.inner.product_rows_pair(compute,weight,x1,x2,rows)}
    fn product_rows_pair_rounded(&self,compute:&C,weight:Weight,x1:&C::Value,x2:&C::Value,rows:usize)->Result<Option<C::Value>> {self.idle("product_rows_pair_rounded")?;self.inner.product_rows_pair_rounded(compute,weight,x1,x2,rows)}
    fn group_product_rows_pair_rounded(&self,compute:&C,weights:&[Weight],pairs:&[(&C::Value,&C::Value)],rows:usize)->Result<Option<C::Value>> {self.idle("group_product_rows_pair_rounded")?;self.inner.group_product_rows_pair_rounded(compute,weights,pairs,rows)}
    fn device_route_group(&self,compute:&C,layer:usize,route:&C::Value,rows:usize,picks:usize,experts:usize)->Result<Option<(C::Value,Self::RoutePlan)>> {
        self.idle("device_route_group")?;
        // Wide decode (`M > 16`, e.g. C8/C16/C32 Auto & C32 Plain) and small prefill (`M <= 256`): shard MoE by
        // expert ID (`0..experts/2` vs `experts/2..experts`) to halve expert DRAM weight reads on both Sparks while
        // preserving exact pick-order `RouteFold` (`0` diffs). Large prefill (`M > 256`) stays on row-split.
        if !self.bypass.get() && rows>spark_gpu_probe::few_product::FEW_MAX && experts>=2
            && (self.decode.get() || rows<=256) && compute.words(route).is_multiple_of(rows)
            && self.pair.arm_route_shard(compute,route,rows,picks,experts)?
        {
            let routed=self.inner.device_route_group(compute,layer,route,rows,picks,experts);
            let slots=self.pair.take_route_shard(compute);
            let Some((weights,full))=routed? else {return Ok(None)};
            if let Some((mine_slots,peer_slots))=slots {
                return Ok(Some((weights,SplitPlan {full,own:None,shard:Some(ShardInfo {rows,picks,mine_slots,peer_slots})})));
            }
            let mut plan=SplitPlan {full,own:None,shard:None};
            if Self::cuts(rows,false) {
                let (own,each)=(self.mine(rows,false),compute.words(route)/rows);
                let own_route=compute.range(route,own.start*each,own.len()*each)?;
                if let Some((own_weights,own_plan))=self.inner.device_route_group(compute,layer,&own_route,own.len(),picks,experts)? {
                    plan.own=Some((own_plan,own_weights,own,rows));
                }
            }
            return Ok(Some((weights,plan)));
        }
        let Some((weights,full))=self.inner.device_route_group(compute,layer,route,rows,picks,experts)? else {return Ok(None)};
        let mut plan=SplitPlan {full,own:None,shard:None};
        if !self.bypass.get() && rows>1 && Self::cuts(rows,false) && compute.words(route).is_multiple_of(rows) {
            let (own,each)=(self.mine(rows,false),compute.words(route)/rows);
            let own_route=compute.range(route,own.start*each,own.len()*each)?;
            if let Some((own_weights,own_plan))=self.inner.device_route_group(compute,layer,&own_route,own.len(),picks,experts)? {plan.own=Some((own_plan,own_weights,own,rows));}
        }
        Ok(Some((weights,plan)))
    }
    fn device_routed_products(&self,compute:&C,first:Weight,route:&Self::RoutePlan,x:&C::Value)->Result<Option<C::Value>> {self.idle("device_routed_products")?;self.inner.device_routed_products(compute,first,&route.full,x)}
    fn device_routed_gate(&self,compute:&C,layer:usize,route:&Self::RoutePlan,x:&C::Value,caps:[f32;4])->Result<Option<C::Value>> {self.idle("device_routed_gate")?;self.inner.device_routed_gate(compute,layer,&route.full,x,caps)}
    fn device_routed_down_fold(&self,compute:&C,layer:usize,route:&Self::RoutePlan,x:&C::Value,weights:&C::Value,picks:usize)->Result<Option<C::Value>> {self.idle("device_routed_down_fold")?;self.inner.device_routed_down_fold(compute,layer,&route.full,x,weights,picks)}
    fn device_routed_experts(&self,compute:&C,layer:usize,route:&Self::RoutePlan,x:&C::Value,caps:[f32;4],weights:&C::Value,picks:usize)->Result<Option<C::Value>> {
        self.idle("device_routed_experts")?;
        if self.bypass.get() {return self.inner.device_routed_experts(compute,layer,&route.full,x,caps,weights,picks);}
        if route.shard.is_some() {return self.shard_cut(compute,layer,route,x,caps,weights);}
        self.row_cut(compute,route,x,|own_plan,own_x,own_weights|self.inner.device_routed_experts(compute,layer,own_plan,own_x,caps,own_weights,picks),|full|self.inner.device_routed_experts(compute,layer,full,x,caps,weights,picks))
    }
    fn device_routed_experts_half_in(&self,compute:&C,layer:usize,route:&Self::RoutePlan,x:&C::Value,caps:[f32;4],weights:&C::Value,picks:usize)->Result<Option<C::Value>> {
        self.idle("device_routed_experts_half_in")?;
        if self.bypass.get() {return self.inner.device_routed_experts_half_in(compute,layer,&route.full,x,caps,weights,picks);}
        // Match 1-Spark's `FEW_MAX` boundary (`M <= 16` in decode uses `picked_products_strided_compact`) for 0-diff
        // bit-exactness, while letting `M > 16` halves (`M = 17..=32` -> `8..=16` per half) stay on `half_in`.
        if self.decode.get() && (picks==0 || compute.words(weights)/picks<=spark_gpu_probe::few_product::FEW_MAX) {return Ok(None);}
        if route.shard.is_some() {return self.shard_cut(compute,layer,route,x,caps,weights);}
        self.row_cut(compute,route,x,|own_plan,own_x,own_weights|{
            let was=<LoadedWeights<'_,V> as Weights<C>>::arithmetic_phase(self.inner,false);
            let out=self.inner.device_routed_experts_half_in(compute,layer,own_plan,own_x,caps,own_weights,picks);
            <LoadedWeights<'_,V> as Weights<C>>::arithmetic_phase(self.inner,was);
            out
        },|full|self.inner.device_routed_experts_half_in(compute,layer,full,x,caps,weights,picks))
    }
    fn device_routed_experts_close(&self,compute:&C,layer:usize,route:&Self::RoutePlan,x:&C::Value,caps:[f32;4],weights:&C::Value,picks:usize,residual:&C::Value,fixed:&[f32],ring:&C::Value,taps:usize,first_slot:usize,bound:f32)->Result<Option<(C::Value,C::Value)>> {
        self.idle("device_routed_experts_close")?;
        // A row or shard cut sums the experts first (`device_routed_experts`); the close then runs on all rows.
        if !self.bypass.get() && (route.own.is_some() || route.shard.is_some()) {return Ok(None);}self.inner.device_routed_experts_close(compute,layer,&route.full,x,caps,weights,picks,residual,fixed,ring,taps,first_slot,bound)
    }
    fn device_routed_experts_close_half_in(&self,compute:&C,layer:usize,route:&Self::RoutePlan,x:&C::Value,caps:[f32;4],weights:&C::Value,picks:usize,residual:&C::Value,fixed:&[f32],ring:&C::Value,taps:usize,first_slot:usize,bound:f32)->Result<Option<(C::Value,C::Value)>> {
        self.idle("device_routed_experts_close_half_in")?;
        // A row or shard cut sums the experts first (`device_routed_experts`); the close then runs on all rows.
        if !self.bypass.get() && (route.own.is_some() || route.shard.is_some()) {return Ok(None);}self.inner.device_routed_experts_close_half_in(compute,layer,&route.full,x,caps,weights,picks,residual,fixed,ring,taps,first_slot,bound)
    }
    fn gate_outer_blend_norm(&self,compute:&C,layer:usize,entries:&[(Message,&C::Value,&C::Value,&C::Value,Option<&[f32]>)],rows:usize,eps:f32,half_out:bool)->Result<Option<Vec<C::Value>>> {
        self.idle("gate_outer_blend_norm")?;self.inner.gate_outer_blend_norm(compute,layer,entries,rows,eps,half_out)
    }
    fn prefers_half_attention(&self,layer:usize)->bool {<LoadedWeights<'_,V> as Weights<C>>::prefers_half_attention(self.inner,layer)}
    fn prefers_half_experts(&self,layer:usize)->bool {<LoadedWeights<'_,V> as Weights<C>>::prefers_half_experts(self.inner,layer)}
    fn prefers_half_messages(&self,start:usize,end:usize)->bool {<LoadedWeights<'_,V> as Weights<C>>::prefers_half_messages(self.inner,start,end)}
    fn prefers_lazy_messages(&self,start:usize,end:usize)->bool {<LoadedWeights<'_,V> as Weights<C>>::prefers_lazy_messages(self.inner,start,end)}
    fn prefill_msg_meta(&self,compute:&C,rows:usize,dim:usize,scale:f32,eps:f32,queries:&[f32],list:&[&C::Value])->Result<Option<(C::Value,C::Value)>> {self.idle("prefill_msg_meta")?;self.inner.prefill_msg_meta(compute,rows,dim,scale,eps,queries,list)}
    fn prefill_gate_fuse(&self,compute:&C,layer:usize,sources:&[&C::Value],entries:&[(Message,&C::Value,&C::Value,&C::Value,&C::Value,Option<&[f32]>)],rows:usize,eps:f32,half_out:bool)->Result<Option<Vec<C::Value>>> {
        self.idle("prefill_gate_fuse")?;self.inner.prefill_gate_fuse(compute,layer,sources,entries,rows,eps,half_out)
    }
    fn prefers_half_output(&self,layer:usize)->bool {<LoadedWeights<'_,V> as Weights<C>>::prefers_half_output(self.inner,layer)}
    fn window_rows_half(&self,compute:&C,a:&crate::bounded_decoder::WindowOne<'_,C::Value>,first_position:u64)->Result<Option<C::Value>> {
        self.idle("window_rows_half")?;
        let align=spark_gpu_probe::flash_window::FW_BKV.max(64);
        if !self.bypass.get() && !self.decode.get() && a.written.is_none()
            && let Some(mid)=window_split_mid(a.rows,a.view,first_position,align)
            && compute.words(a.query).is_multiple_of(a.rows)
            && compute.words(a.moved).is_multiple_of(a.rows)
            && compute.words(a.sink).is_multiple_of(a.rows)
            && compute.words(a.gate).is_multiple_of(a.rows)
            && compute.words(a.steps)==2*a.rows
            && a.position.is_none_or(|p|compute.words(p)==a.rows)
        {
            let own=[0..mid,mid..a.rows][self.pair.half()].clone();
            let (q_w,m_w,s_w,g_w)=(compute.words(a.query)/a.rows,compute.words(a.moved)/a.rows,compute.words(a.sink)/a.rows,compute.words(a.gate)/a.rows);
            let q=compute.range(a.query,own.start*q_w,own.len()*q_w)?;
            let m=compute.range(a.moved,own.start*m_w,own.len()*m_w)?;
            let s=compute.range(a.sink,own.start*s_w,own.len()*s_w)?;
            let g=compute.range(a.gate,own.start*g_w,own.len()*g_w)?;
            let st=compute.range(a.steps,own.start*2,own.len()*2)?;
            let pos=a.position.map(|p|compute.range(p,own.start,own.len())).transpose()?;
            let sub=crate::bounded_decoder::WindowOne {
                rows:own.len(),query:&q,moved:&m,sink:&s,gate:&g,steps:&st,position:pos.as_ref(),..*a
            };
            if let Some(local)=self.inner.window_rows_half(compute,&sub,first_position+own.start as u64)? {
                let row_words=compute.words(&local)/own.len();
                return self.whole(compute,&local,1,(a.rows-own.len())*row_words).map(Some);
            }
        }
        self.inner.window_rows_half(compute,a,first_position)
    }
    fn mix_rows_normed_half(&self,compute:&C,rows:usize,dim:usize,scale:f32,eps:f32,post_eps:f32,transpose:bool,keep_normed:bool,keep_mixed:bool,queries:&[f32],list:&[&C::Value],stream:&C::Value,post_weight:Option<&[f32]>,post_weight_per_query:bool)->Result<Option<(Option<C::Value>,Option<C::Value>,C::Value)>> {
        self.idle("mix_rows_normed_half")?;self.inner.mix_rows_normed_half(compute,rows,dim,scale,eps,post_eps,transpose,keep_normed,keep_mixed,queries,list,stream,post_weight,post_weight_per_query)
    }
    fn mix_prefill_normed_half(&self,compute:&C,rows:usize,dim:usize,scale:f32,eps:f32,queries:&[f32],list:&[&C::Value])->Result<Option<(C::Value,C::Value)>> {self.idle("mix_prefill_normed_half")?;self.inner.mix_prefill_normed_half(compute,rows,dim,scale,eps,queries,list)}
    fn product_rows_half_in(&self,compute:&C,weight:Weight,x:&C::Value,rows:usize)->Result<Option<C::Value>> {self.idle("product_rows_half_in")?;self.inner.product_rows_half_in(compute,weight,x,rows)}
    fn stack_products_rows_half_in(&self,compute:&C,weights:&[Weight],x:&C::Value,rows:usize)->Result<Option<Vec<C::Value>>> {
        self.idle("stack_products_rows_half_in")?;
        // Wide decode query stack (`M > 16`, e.g. C32 plain & batched drafted verify): partition at a band-aligned
        // whole-matrix boundary `k` (`stack_split`), pairing trailing K/V onto opposite halves when it improves row
        // balance (`4480` vs `4120` instead of `3456` vs `5144`), and trade in one contiguous `parts == 1` block.
        if !self.bypass.get() && self.decode.get() && rows>spark_gpu_probe::few_product::FEW_MAX && Self::query_stack(weights)
            && let Some((_,_,parts,columns))=self.inner.run(weights)?
            && let Some(k)=stack_split(&parts)
            && columns.is_multiple_of(2) && compute.words(x)==rows*(columns/2)
        {
            let left:usize=parts[..k].iter().sum();
            let right:usize=parts[k..].iter().sum();
            let tail=weights.len().saturating_sub(2);
            if weights.len()>=k+3 {
                let mid_rows:usize=parts[k..tail].iter().sum();
                let (pa,pb)=(parts[tail],parts[tail+1]);
                if (left+pa).abs_diff(mid_rows+pb)<left.abs_diff(right)
                    && spark_gpu_probe::tc_rows::tc_shape_admitted(left,columns,rows)
                    && spark_gpu_probe::tc_rows::tc_shape_admitted(mid_rows,columns,rows)
                    && spark_gpu_probe::tc_rows::tc_shape_admitted(pa,columns,rows)
                    && spark_gpu_probe::tc_rows::tc_shape_admitted(pb,columns,rows)
                    && self.inner.run(&weights[..k])?.is_some()
                    && self.inner.run(&weights[k..tail])?.is_some()
                {
                    let (mine_w,extra_w,other_rows)=if self.pair.half()==0 {(&weights[..k],weights[tail],mid_rows+pb)} else {(&weights[k..tail],weights[tail+1],left+pa)};
                    if let (Some(mut mine),Some(extra))=(self.inner.stack_products_rows_half_in(compute,mine_w,x,rows)?,self.inner.product_rows_half_in(compute,extra_w,x,rows)?) {
                        mine.push(extra);
                        let refs:Vec<&C::Value>=mine.iter().collect();
                        let whole=self.whole_parts(compute,&refs,rows*other_rows)?;
                        drop(mine);
                        let mut out=Vec::with_capacity(parts.len());
                        let mut at=0;
                        for &size in &parts[..k] {out.push(compute.range(&whole,at,rows*size)?);at+=rows*size;}
                        let part_a=compute.range(&whole,at,rows*pa)?;at+=rows*pa;
                        for &size in &parts[k..tail] {out.push(compute.range(&whole,at,rows*size)?);at+=rows*size;}
                        let part_b=compute.range(&whole,at,rows*pb)?;
                        out.push(part_a);out.push(part_b);
                        return Ok(Some(out));
                    }
                }
            }
            if spark_gpu_probe::tc_rows::tc_shape_admitted(left,columns,rows)
                && spark_gpu_probe::tc_rows::tc_shape_admitted(right,columns,rows)
                && self.inner.run(&weights[..k])?.is_some()
                && self.inner.run(&weights[k..])?.is_some()
            {
                let (mine_w,other_rows)=if self.pair.half()==0 {(&weights[..k],right)} else {(&weights[k..],left)};
                if let Some(mine)=self.inner.stack_products_rows_half_in(compute,mine_w,x,rows)? {
                    let refs:Vec<&C::Value>=mine.iter().collect();
                    let whole=self.whole_parts(compute,&refs,rows*other_rows)?;
                    drop(mine);
                    let mut start=0;
                    return parts.iter().map(|&size|{let p=compute.range(&whole,start,rows*size);start+=rows*size;p}).collect::<Result<Vec<_>>>().map(Some);
                }
            }
        }
        self.inner.stack_products_rows_half_in(compute,weights,x,rows)
    }
    fn product_rows_pair_half_in(&self,compute:&C,weight:Weight,x1:&C::Value,x2:&C::Value,rows:usize)->Result<Option<C::Value>> {self.idle("product_rows_pair_half_in")?;self.inner.product_rows_pair_half_in(compute,weight,x1,x2,rows)}
    fn product_runs(&self,compute:&C,runs:&[(Weight,usize)],x:&C::Value,columns:usize,input:Option<&[usize]>,output:Option<&[usize]>)->Result<C::Value> {self.idle("product_runs")?;self.inner.product_runs(compute,runs,x,columns,input,output)}
}
pub mod mirror;
pub mod serve;
#[cfg(test)]
mod tests;
