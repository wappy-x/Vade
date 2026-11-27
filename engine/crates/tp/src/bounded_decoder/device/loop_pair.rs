//! The trade of a cut decode product between the two Sparks of a cabled pair (`loop_decoder::tp2`), on the ordered
//! stream of the device: one joined put (the message to the other Spark, and this Spark's part at its place in the
//! whole result) and one take (the other Spark's part into its place). The host neither copies nor waits; the wire
//! thread of the mailbox (`tensor_wire`) moves each message with one-sided writes. Every word of the result is a
//! word of one Spark's product: the trade moves words and computes none.
use super::*;
use spark_gpu_probe::driver::TpMailbox;

impl<'b,'d> Device<'b,'d> {
    /// The whole result of a cut product from this Spark's part `local` (`parts` rows of the same count of words) and
    /// the other Spark's part (`parts` rows of `theirs` words), mailbox message `seq`: `[parts][own + theirs]`, in each
    /// row the lower part first (half 0 holds the lower part). The take waits at most `limit_ns` for the message.
    /// None, with nothing queued: the message does not fit one slot of `m`, or one joined put does not take its
    /// shape (`tp_put_join::admitted`); the caller trades on another wire.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pair_trade(&self,m:&TpMailbox<'d>,local:&Value<'b,'d>,parts:usize,theirs:usize,half:usize,seq:u64,limit_ns:u64)->Result<Option<Value<'b,'d>>> {
        if parts==0 || !local.words.is_multiple_of(parts) || half>1 || local.words==0 || theirs==0 {return Err("pair trade: the parts hold no whole rows".into());}
        let own=local.words/parts;let width=own+theirs;
        let slot=m.slot_bytes()/4;
        if local.words>slot || parts*theirs>slot {return Ok(None);}
        self.ready(local)?;
        let (mine_at,peer_at)=[(0,own),(theirs,0)][half];
        let mut out=self.backend.vector(parts*width)?;
        let rows=(0..parts).map(|row|[row*own,row*width+mine_at,own]).collect::<Vec<_>>();
        if !self.backend.tp_mailbox_put_join(m,&[(&local.vector,false)],&mut out,&rows,seq)? {return Ok(None);}
        self.backend.tp_mailbox_take(m,&mut out,parts,width,theirs,peer_at,seq,limit_ns)?;
        self.value(out,parts*width).map(Some)
    }
}

impl<'b,'d> Device<'b,'d> {
    /// A value of `words` words that no launch writes or reads: the place of a cut product whose words this Spark
    /// keeps elsewhere (`loop_decoder::tp2::Pair::blank`). No launch fills it.
    pub(crate) fn pair_blank(&self,words:usize)->Result<Value<'b,'d>> {
        let out=self.backend.vector(words.max(1))?;self.value(out,words)
    }
    /// A fresh result of `words` words for a joined trade (`pair_trade_into`) whose longest part is `most` words
    /// of at most `parts` products: its words are written only by that trade, which the caller queues before any
    /// launch reads them. None: one slot of `m` or one joined put does not take that shape.
    pub(crate) fn pair_open(&self,m:&TpMailbox<'d>,words:usize,most:usize,parts:usize)->Result<Option<Value<'b,'d>>> {
        if words==0 || most>m.slot_bytes()/4 || !spark_gpu_probe::tp_put_join::admitted(parts,1,most,words) {return Ok(None);}
        let out=self.backend.vector(words)?;self.value(out,words).map(Some)
    }
    /// The joined trade of one cut whose part on this Spark is the products `locals` one after another: they go to
    /// words `mine_at..` of `whole` (one row) and to the other Spark, and the other Spark's part (`theirs` words)
    /// comes into words `peer_at..`. False, with nothing queued: one joined put does not take this shape.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pair_trade_into(&self,m:&TpMailbox<'d>,locals:&[&Value<'b,'d>],whole:&mut Value<'b,'d>,mine_at:usize,peer_at:usize,theirs:usize,seq:u64,limit_ns:u64)->Result<bool> {
        let own=locals.iter().map(|v|v.words).sum::<usize>();let width=whole.words;
        if own==0 || theirs==0 || own+theirs!=width || own>m.slot_bytes()/4 || theirs>m.slot_bytes()/4 {return Ok(false);}
        if !spark_gpu_probe::tp_put_join::admitted(locals.len(),1,own,width) {return Ok(false);}
        for local in locals {self.ready(local)?;}
        let Some(out)=std::rc::Rc::get_mut(&mut whole.vector) else {return Err("pair trade: the joined result is shared before its trade".into())};
        let parts=locals.iter().map(|v|(&*v.vector,false)).collect::<Vec<_>>();
        if !self.backend.tp_mailbox_put_join(m,&parts,out,&[[0,mine_at,own]],seq)? {return Ok(false);}
        self.backend.tp_mailbox_take(m,out,1,width,theirs,peer_at,seq,limit_ns)?;
        Ok(true)
    }
    /// The whole result of a cut (as `pair_trade`: `parts` rows of this Spark's part `local`, the other part
    /// `theirs` words a row) for a part that one joined put does not take: one put of this Spark's part as it is,
    /// one copy of its rows to their places, one take of the other part. The same words as `pair_trade`. None, with
    /// nothing queued: a part does not fit one slot.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pair_trade_block(&self,m:&TpMailbox<'d>,local:&Value<'b,'d>,parts:usize,theirs:usize,half:usize,seq:u64,limit_ns:u64)->Result<Option<Value<'b,'d>>> {
        let slot=m.slot_bytes()/4;
        if parts==0 || local.words==0 || theirs==0 || half>1 || !local.words.is_multiple_of(parts) || local.words>slot || parts*theirs>slot || local.vector.is_segmented() {return Ok(None);}
        self.ready(local)?;
        let own=local.words/parts;let width=own+theirs;let (mine_at,peer_at)=[(0,own),(theirs,0)][half];
        let mut out=self.backend.vector(parts*width)?;
        self.backend.tp_mailbox_put_finite(m,&local.vector,1,local.words,local.words,0,seq)?;
        if parts==1 {self.backend.copy_range(&local.vector,&mut out,0,mine_at,own)?;}
        else {self.backend.resident_scatter_copy(&local.vector,&mut out,&(0..parts).map(|row|[row*own,row*width+mine_at,own]).collect::<Vec<_>>())?;}
        self.backend.tp_mailbox_take(m,&mut out,parts,width,theirs,peer_at,seq,limit_ns)?;
        self.value(out,parts*width).map(Some)
    }
    /// Dual-rail block trade over two independent RDMA mailboxes (`m0`, `m1` on separate PCIe rails): splits the
    /// payload in half across the two mailboxes so both rails transfer simultaneously, overlapping with the local
    /// copy. None, with nothing queued: shape outside dual-rail alignment/slot bounds (caller falls back to 1 rail).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pair_trade_block_dual(
        &self, m0:&TpMailbox<'d>, m1:&TpMailbox<'d>, local:&Value<'b,'d>, parts:usize, theirs:usize, half:usize,
        seq0:u64, seq1:u64, limit_ns:u64,
    )->Result<Option<Value<'b,'d>>> {
        let (slot0,slot1)=(m0.slot_bytes()/4,m1.slot_bytes()/4);
        if parts==0 || local.words==0 || theirs==0 || half>1 || !local.words.is_multiple_of(parts) || local.vector.is_segmented() {return Ok(None);}
        let own=local.words/parts;let width=own+theirs;let (mine_at,peer_at)=[(0,own),(theirs,0)][half];
        if parts==1 {
            let own0=(own/2)&!3;let own1=own-own0;
            let peer0=(theirs/2)&!3;let peer1=theirs-peer0;
            if own0==0 || own1==0 || peer0==0 || peer1==0 || !own1.is_multiple_of(4) || !peer1.is_multiple_of(4)
                || own0>slot0 || own1>slot1 || peer0>slot0 || peer1>slot1 {return Ok(None);}
            self.ready(local)?;
            let mut out=self.backend.vector(width)?;
            self.backend.tp_mailbox_put_finite(m0,&local.vector,1,own,own0,0,seq0)?;
            self.backend.tp_mailbox_put_finite(m1,&local.vector,1,own,own1,own0,seq1)?;
            self.backend.copy_range(&local.vector,&mut out,0,mine_at,own)?;
            self.backend.tp_mailbox_take(m0,&mut out,1,width,peer0,peer_at,seq0,limit_ns)?;
            self.backend.tp_mailbox_take(m1,&mut out,1,width,peer1,peer_at+peer0,seq1,limit_ns)?;
            return self.value(out,width).map(Some);
        }
        let p0=parts/2;let p1=parts-p0;
        let (own0,own1,peer0,peer1)=(p0*own,p1*own,p0*theirs,p1*theirs);
        if p0==0 || p1==0 || p1>65535 || !own0.is_multiple_of(4) || !own1.is_multiple_of(4) || !peer0.is_multiple_of(4) || !peer1.is_multiple_of(4)
            || own0>slot0 || own1>slot1 || peer0>slot0 || peer1>slot1 {return Ok(None);}
        self.ready(local)?;
        let mut out=self.backend.vector(parts*width)?;
        self.backend.tp_mailbox_put_finite(m0,&local.vector,1,local.words,own0,0,seq0)?;
        self.backend.tp_mailbox_put_finite(m1,&local.vector,1,local.words,own1,own0,seq1)?;
        self.backend.resident_scatter_copy(&local.vector,&mut out,&(0..parts).map(|row|[row*own,row*width+mine_at,own]).collect::<Vec<_>>())?;
        self.backend.tp_mailbox_take(m0,&mut out,p0,width,theirs,peer_at,seq0,limit_ns)?;
        let tail_rows=(0..p1).map(|row|[row*theirs,(p0+row)*width+peer_at,theirs]).collect::<Vec<_>>();
        self.backend.tp_mailbox_take_scatter(m1,&mut out,&tail_rows,0,seq1,limit_ns)?;
        self.value(out,parts*width).map(Some)
    }
    /// Contiguous multi-part block trade (`parts == 1`) over one or two RDMA mailboxes: copies `locals` directly
    /// into `out[mine_at..]` without an intermediate allocation or segmented `DeviceVector`, then puts from `out`
    /// and takes peer's `theirs` words into `out[peer_at..]`. Uses dual-rail when `m1` is `Some` and aligned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pair_trade_parts(
        &self, m0:&TpMailbox<'d>, m1:Option<&TpMailbox<'d>>, locals:&[&Value<'b,'d>], theirs:usize, half:usize,
        seq0:u64, seq1:u64, limit_ns:u64,
    )->Result<Option<(Value<'b,'d>,bool)>> {
        if locals.is_empty() || theirs==0 || half>1 || locals.iter().any(|v|v.words==0 || v.vector.is_segmented()) {return Ok(None);}
        let own:usize=locals.iter().map(|v|v.words).sum();
        let width=own.checked_add(theirs).ok_or("pair trade parts: width overflow")?;
        let (mine_at,peer_at)=[(0,own),(theirs,0)][half];
        let slot0=m0.slot_bytes()/4;
        if let Some(m1)=m1 {
            let slot1=m1.slot_bytes()/4;
            let own0=(own/2)&!3;let own1=own-own0;
            let peer0=(theirs/2)&!3;let peer1=theirs-peer0;
            if own0>0 && own1>0 && peer0>0 && peer1>0 && own1.is_multiple_of(4) && peer1.is_multiple_of(4)
                && mine_at.is_multiple_of(4)
                && own0<=slot0 && own1<=slot1 && peer0<=slot0 && peer1<=slot1
            {
                for v in locals {self.ready(v)?;}
                let mut out=self.backend.vector(width)?;
                let mut off=mine_at;
                for v in locals {self.backend.copy_range(&v.vector,&mut out,0,off,v.words)?;off+=v.words;}
                self.backend.tp_mailbox_put_finite(m0,&out,1,width,own0,mine_at,seq0)?;
                self.backend.tp_mailbox_put_finite(m1,&out,1,width,own1,mine_at+own0,seq1)?;
                self.backend.tp_mailbox_take(m0,&mut out,1,width,peer0,peer_at,seq0,limit_ns)?;
                self.backend.tp_mailbox_take(m1,&mut out,1,width,peer1,peer_at+peer0,seq1,limit_ns)?;
                return Ok(Some((self.value(out,width)?,true)));
            }
        }
        if own>slot0 || theirs>slot0 {return Ok(None);}
        for v in locals {self.ready(v)?;}
        let mut out=self.backend.vector(width)?;
        if spark_gpu_probe::tp_put_join::admitted(locals.len(),1,own,width) {
            let parts=locals.iter().map(|v|(&*v.vector,false)).collect::<Vec<_>>();
            if self.backend.tp_mailbox_put_join(m0,&parts,&mut out,&[[0,mine_at,own]],seq0)? {
                self.backend.tp_mailbox_take(m0,&mut out,1,width,theirs,peer_at,seq0,limit_ns)?;
                return Ok(Some((self.value(out,width)?,false)));
            }
        }
        let mut off=mine_at;
        for v in locals {self.backend.copy_range(&v.vector,&mut out,0,off,v.words)?;off+=v.words;}
        self.backend.tp_mailbox_put_finite(m0,&out,1,width,own,mine_at,seq0)?;
        self.backend.tp_mailbox_take(m0,&mut out,1,width,theirs,peer_at,seq0,limit_ns)?;
        Ok(Some((self.value(out,width)?,false)))
    }
    /// Pack `[words]` BF16-valued FP32 words into `[words / 2]` packed BF16 words via vectorized `PREFILL_OP_PACK`
    /// inside the resident stream (zero graph flush, bit-exact `cvt.rn.bf16x2.f32`).
    pub(crate) fn pair_pack_bf16(&self,x:&Value<'b,'d>)->Result<Option<Value<'b,'d>>> {
        if x.words<8 || !x.words.is_multiple_of(8) || x.vector.is_segmented() || x.vector.device_address()&15!=0 || !self.backend.resident_launches_active() {
            return Ok(None);
        }
        self.ready(x)?;
        let packed=self.backend.prefill_pack_bf16(1,x.words,&x.vector)?;
        self.value(packed,x.words/2).map(Some)
    }
    /// Unpack `[rows * dim / 2]` packed BF16 words back into `[rows * dim]` FP32 words via resident
    /// `Kind::FastStreamOps` (`fast_route_fold_half_into`, `picks = 1`, `weights = 1.0`): zero graph flush and full
    /// coalesced DRAM bandwidth, matching `cvt.rn.bf16x2.f32` lane order bit-for-bit on `rn`-rounded sums.
    pub(crate) fn pair_unpack_bf16(&self,x:&Value<'b,'d>,rows:usize,dim:usize)->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::fast_ops;
        if rows==0 || dim==0 || !dim.is_multiple_of(2) || x.words!=rows*(dim/2) || x.vector.is_segmented()
            || !self.backend.resident_launches_active()
            || !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD,rows,dim,1,fast_ops::STREAM_OPS_THREADS)
        {
            return Ok(None);
        }
        self.ready(x)?;
        let ones=vec![1.0f32;rows];
        let weights=self.parameters.borrow_mut().resolve(&ones,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("pair unpack bf16: missing unit weights")?;
        let mut out=self.backend.vector(rows*dim)?;
        self.backend.fast_route_fold_half_into(rows,1,dim,&weights,&x.vector,&mut out)?;
        self.value(out,rows*dim).map(Some)
    }
    /// Arm 2-Spark expert-ID sharding (`0..experts/2` on half 0, `experts/2..experts` on half 1) for the
    /// immediately following `device_route_group` call on `route`. 1-Spark never calls this.
    pub(crate) fn pair_arm_route_shard(&self,route:&Value<'b,'d>,rows:usize,picks:usize,experts:usize,half:usize)->Result<bool> {
        if rows==0 || picks==0 || experts<2 || half>1 || route.vector.is_segmented() || !self.backend.resident_launches_active() {
            return Ok(false);
        }
        self.ready(route)?;
        self.backend.arm_moe_route_shard(&route.vector,rows,picks,experts,half);
        Ok(true)
    }
    /// Consume `(mine_slots, peer_slots)` produced by an armed sharded `device_route_group` (clearing any armed state).
    pub(crate) fn pair_take_route_shard(&self)->Option<(Vec<usize>,Vec<usize>)> {
        self.backend.take_moe_route_shard()
    }
    /// Dual-rail RDMA exchange + direct pick-order scatter + bit-exact `PREFILL_OP_FOLD` for expert-ID sharded MoE.
    /// Each half sends its contiguous BF16 down outputs (`mine_slots.len()` rows of `dim / 2` u32 words), scatters
    /// its own rows directly into pick-order `full_downs` while RDMA is in flight, scatters peer's rows directly from
    /// the receive slots into `full_downs`, and folds all `picks` per token in exact original pick order `0..picks`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pair_trade_shard_fold(
        &self, m0:&TpMailbox<'d>, m1:Option<&TpMailbox<'d>>, local_bf16:&Value<'b,'d>, weights:&Value<'b,'d>,
        rows:usize, picks:usize, dim:usize, mine_slots:&[usize], peer_slots:&[usize],
        seq0:&std::cell::Cell<u64>, seq1:&std::cell::Cell<u64>, limit_ns:u64,
    )->Result<Option<Value<'b,'d>>> {
        let all=rows.checked_mul(picks).ok_or("pair shard fold: pick overflow")?;
        let (own_picks,peer_picks)=(mine_slots.len(),peer_slots.len());
        if rows==0 || picks==0 || dim==0 || !dim.is_multiple_of(2) || own_picks==0 || peer_picks==0 || own_picks+peer_picks!=all {
            return Ok(None);
        }
        let row_words=dim/2;
        if local_bf16.words!=own_picks*row_words || weights.words!=all
            || local_bf16.vector.is_segmented() || weights.vector.is_segmented()
            || !self.backend.resident_launches_active()
        {
            return Ok(None);
        }
        self.ready(local_bf16)?;
        self.ready(weights)?;
        let (own_words,peer_words)=(own_picks*row_words,peer_picks*row_words);
        let slot0=m0.slot_bytes()/4;
        if row_words>slot0 {return Ok(None);}
        let mut full_downs=self.backend.vector(all*row_words)?;
        self.backend.skip_fill(&mut full_downs)?;
        let mine_rows:Vec<[usize;3]>=mine_slots.iter().enumerate()
            .map(|(i,&slot)|[i*row_words,slot*row_words,row_words]).collect();
        let (own_p0,own_p1)=(own_picks/2,own_picks-own_picks/2);
        let (peer_p0,peer_p1)=(peer_picks/2,peer_picks-peer_picks/2);
        let dual=m1.filter(|m1|{
            let slot1=m1.slot_bytes()/4;
            own_p0>0 && own_p1>0 && peer_p0>0 && peer_p1>0
                && own_p0*row_words<=slot0 && own_p1*row_words<=slot1
                && peer_p0*row_words<=slot0 && peer_p1*row_words<=slot1
                && (own_p0*row_words).is_multiple_of(4) && (own_p1*row_words).is_multiple_of(4)
                && (peer_p0*row_words).is_multiple_of(4) && (peer_p1*row_words).is_multiple_of(4)
        });
        if let Some(m1)=dual {
            let (s0,s1)=(seq0.get()+1,seq1.get()+1);
            let mut peer_buf=self.backend.vector(peer_words)?;
            self.backend.skip_fill(&mut peer_buf)?;
            self.backend.tp_mailbox_put_finite(m0,&local_bf16.vector,1,own_words,own_p0*row_words,0,s0)?;
            self.backend.tp_mailbox_put_finite(m1,&local_bf16.vector,1,own_words,own_p1*row_words,own_p0*row_words,s1)?;
            self.backend.resident_scatter_copy(&local_bf16.vector,&mut full_downs,&mine_rows)?;
            // Contiguous dual-rail `take` into device `peer_buf` (~20 us/rail) + fast D2D scatter (~24 us),
            // avoiding 256-CTA zero-copy `take_scatter` overhead (~150 us/layer).
            self.backend.tp_mailbox_take(m0,&mut peer_buf,1,peer_words,peer_p0*row_words,0,s0,limit_ns)?;
            self.backend.tp_mailbox_take(m1,&mut peer_buf,1,peer_words,peer_p1*row_words,peer_p0*row_words,s1,limit_ns)?;
            let peer_rows:Vec<[usize;3]>=peer_slots.iter().enumerate()
                .map(|(j,&slot)|[j*row_words,slot*row_words,row_words]).collect();
            self.backend.resident_scatter_copy(&peer_buf,&mut full_downs,&peer_rows)?;
            seq0.set(s0);seq1.set(s1);
        } else if own_words<=slot0 && peer_words<=slot0 {
            let s0=seq0.get()+1;
            let mut peer_buf=self.backend.vector(peer_words)?;
            self.backend.skip_fill(&mut peer_buf)?;
            self.backend.tp_mailbox_put_finite(m0,&local_bf16.vector,1,own_words,own_words,0,s0)?;
            self.backend.resident_scatter_copy(&local_bf16.vector,&mut full_downs,&mine_rows)?;
            self.backend.tp_mailbox_take(m0,&mut peer_buf,1,peer_words,peer_words,0,s0,limit_ns)?;
            let peer_rows:Vec<[usize;3]>=peer_slots.iter().enumerate()
                .map(|(j,&slot)|[j*row_words,slot*row_words,row_words]).collect();
            self.backend.resident_scatter_copy(&peer_buf,&mut full_downs,&peer_rows)?;
            seq0.set(s0);
        } else {
            // Chunked mailbox exchange when a huge batch exceeds one slot (zero arbitrary caps, Rule 17).
            let chunk=(slot0/row_words).max(1);
            let rounds=own_picks.max(peer_picks).div_ceil(chunk);
            self.backend.resident_scatter_copy(&local_bf16.vector,&mut full_downs,&mine_rows)?;
            for r in 0..rounds {
                let s0=seq0.get()+1;
                let o_start=(r*chunk).min(own_picks);
                let o_cnt=(own_picks-o_start).min(chunk);
                let p_start=(r*chunk).min(peer_picks);
                let p_cnt=(peer_picks-p_start).min(chunk);
                if o_cnt>0 {
                    self.backend.tp_mailbox_put_finite(m0,&local_bf16.vector,1,own_words,o_cnt*row_words,o_start*row_words,s0)?;
                } else {
                    self.backend.tp_mailbox_put_finite(m0,&local_bf16.vector,1,own_words,1,0,s0)?;
                }
                if p_cnt>0 {
                    let pr:Vec<[usize;3]>=peer_slots[p_start..p_start+p_cnt].iter().enumerate()
                        .map(|(j,&slot)|[j*row_words,slot*row_words,row_words]).collect();
                    self.backend.tp_mailbox_take_scatter(m0,&mut full_downs,&pr,0,s0,limit_ns)?;
                } else {
                    let mut dummy=self.backend.vector(1)?;
                    self.backend.tp_mailbox_take(m0,&mut dummy,1,1,1,0,s0,limit_ns)?;
                }
                seq0.set(s0);
            }
        }
        let mut output=self.backend.vector(rows*dim)?;
        if dim.is_multiple_of(8) && ((full_downs.device_address()|output.device_address())&15)==0 {
            self.backend.prefill_route_fold_half_into(rows,picks,dim,&weights.vector,&full_downs,&mut output)?;
        } else {
            self.backend.fast_route_fold_half_into(rows,picks,dim,&weights.vector,&full_downs,&mut output)?;
        }
        self.value(output,rows*dim).map(Some)
    }
}
