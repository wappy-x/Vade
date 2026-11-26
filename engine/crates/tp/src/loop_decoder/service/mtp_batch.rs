//! Multi-chat batched DSpark drafting, cohort-aware confidence gating, and batched speculative verify (`C2..=C32`).
//!
//! - `VERIFY_ROW_CAP` (`256`) fits `C8 d<=7` (`M<=64`), `C16 d<=7` (`M<=128`), and `C32 d<=7` (`M<=256`) in ONE
//!   single-pass grouped `PLAIN_PACKED_TC_ROWS` target verify with `short_conv_multi` (`rows <= 256`).
//! - `DRAFT_CHAT_LIMIT` (`32`) matches `DRAFT_BATCH_MAX` so `C2..=C32` all share ONE batched DSpark draft pass.
use super::{BatchRows,Chat,ChatStep,Piece,Predictions,Service,Slot,Wanted,mtp::{Context,LAG_ROWS,Model,SHALLOW_PROPOSE},need};
use crate::{Result,bounded_decoder::Compute,loop_decoder::{experts::Phase,mix::Weights}};
use spark_engine::draft::ModelProposal;
use spark_serve::{BatchError,BatchOutput,DraftRows};
use std::time::Instant;

/// Maximum verify rows per single target pass inside the wide resident decode & `short_conv_multi` envelope.
pub(super) const VERIFY_ROW_CAP:usize=256;
/// Largest concurrent cohort that drafts (`<= VERIFY_ROW_CAP / 2` and `<= DRAFT_BATCH_MAX`).
pub(super) const DRAFT_CHAT_LIMIT:usize=32;

/// Minimum cohort rounds before checking sustained negative return vs plain cohort decode.
const COHORT_FALLBACK_MIN_ROUNDS:u32=24;
const COHORT_PROBE_FIRST:usize=8;
const COHORT_PROBE_MOST:usize=64;

/// Maximum uniform propose depth per concurrency band:
/// - `C2..=C8`: up to `7` (`M <= 64`, fits the graph cache with zero evictions and `< 6 ms` draft).
/// - `C9..=C32`: up to `3` (`M <= 64` at `C16`, `M <= 128` at `C32`), keeping draft `< 9 ms`, graph cache
///   eviction-free (`2` verify widths), and straggler spread tight across diverse cohorts.
fn max_cohort_depth(chats:usize)->usize {if chats>8 {3} else {7}}

struct CohortState {
    ids:Vec<u64>,
    plain_ms:Option<f64>,
    plain_n:u32,
    verify_ms:Vec<Option<f64>>,
    verify_n:Vec<u32>,
    tokens:f64,
    ms:f64,
    rounds:u32,
    paused:bool,
    plain_run:usize,
    gap:usize,
    probing:bool,
}

impl Default for CohortState {
    fn default()->Self {
        Self {
            ids:Vec::new(),
            plain_ms:None,plain_n:0,
            verify_ms:vec![None;VERIFY_ROW_CAP+1],verify_n:vec![0;VERIFY_ROW_CAP+1],
            tokens:0.0,ms:0.0,rounds:0,
            paused:false,plain_run:0,gap:COHORT_PROBE_FIRST,probing:false,
        }
    }
}

struct OpenRound {
    id:u64,chats:usize,ids:Vec<u64>,round_ms:f64,remaining:usize,kept:usize,
}

fn sorted_ids(iter:impl Iterator<Item=u64>)->Vec<u64> {
    let mut ids:Vec<u64>=iter.collect();
    ids.sort_unstable();
    ids
}

/// Measured multi-chat step timings and cohort-level rate calibration, scoped per active cohort of chat IDs so
/// distinct prompt cohorts (identical vs diverse MoE expert mixes, structured vs prose) never poison each other's
/// `plain_ms`, `verify_ms`, `target_rate`, or `paused` state. Keeps `C1` single-chat `Prices` untouched.
pub(super) struct CohortPricing {
    cohorts:Vec<CohortState>,
    draft_ms:Vec<Option<f64>>,
    draft_n:Vec<u32>,
    push_ms:Option<f64>,
    last_draft_ms:f64,
    next_round_id:u64,
    open:Option<OpenRound>,
}

impl Default for CohortPricing {
    fn default()->Self {
        Self {
            cohorts:(0..=DRAFT_CHAT_LIMIT).map(|_|CohortState::default()).collect(),
            draft_ms:vec![None;DRAFT_CHAT_LIMIT+1],
            draft_n:vec![0;DRAFT_CHAT_LIMIT+1],
            push_ms:None,
            last_draft_ms:0.0,next_round_id:1,open:None,
        }
    }
}

impl CohortPricing {
    fn ema(slot:&mut Option<f64>,n:&mut u32,ms:f64) {
        *n=n.saturating_add(1);
        *slot=Some(match (*slot,*n) {
            (None,_)|(Some(_),1)=>ms,
            // Replace a cold first-run CUDA graph capture spike immediately when sample 2 is faster.
            (Some(old),2) if ms<old=>ms,
            (Some(old),_)=>if ms<old {0.35*old+0.65*ms} else {0.75*old+0.25*ms},
        });
    }
    fn state_mut(&mut self,chats:usize,ids:&[u64])->Option<&mut CohortState> {
        if self.cohorts.get(chats)?.ids.as_slice()!=ids {
            // When a cohort shrinks as chats finish (`ids` is a strict subset of a live larger cohort), seed `plain_ms`
            // from the parent cohort so the remaining straggler chats keep drafting immediately without wasting plain
            // calibration ticks. Disjoint cohorts always start fresh and isolated.
            let parent=(chats+1..=DRAFT_CHAT_LIMIT).find_map(|k| {
                let s=&self.cohorts[k];
                let p_ms=s.plain_ms?;
                (!s.ids.is_empty() && ids.iter().all(|id|s.ids.contains(id)))
                    .then(||(p_ms*(chats as f64/k as f64).powf(0.45),s.paused))
            });
            let (plain_ms,paused)=parent.map_or((None,false),|(ms,p)|(Some(ms),p));
            self.cohorts[chats]=CohortState {ids:ids.to_vec(),plain_ms,paused,..CohortState::default()};
        }
        self.cohorts.get_mut(chats)
    }
    fn state_ref(&self,chats:usize,ids:&[u64])->Option<&CohortState> {
        self.cohorts.get(chats).filter(|s|s.ids.as_slice()==ids)
    }
    fn note_plain(&mut self,chats:usize,ids:&[u64],ms:f64) {
        if let Some(state)=self.state_mut(chats,ids) {
            Self::ema(&mut state.plain_ms,&mut state.plain_n,ms);
        }
    }
    /// Record one verify step timing. Ignores the cold first-ever run (`verify_n == 1`, which includes one-time
    /// CUDA graph capture and vector-pool allocation) and smoothly clamps upward jumps (`<= 1.45x`) so single-tick
    /// jitter never spikes `step_at` while genuine steady-state MoE divergence still converges cleanly.
    fn note_verify(&mut self,chats:usize,ids:&[u64],rows:usize,ms:f64)->bool {
        let Some(state)=self.state_mut(chats,ids) else {return false;};
        let (Some(slot),Some(n))=(state.verify_ms.get_mut(rows),state.verify_n.get_mut(rows)) else {return false;};
        *n=n.saturating_add(1);
        if *n<2 {return false;}
        if let Some(old)=*slot {
            let clamped=ms.min(old*1.45);
            *slot=Some(if clamped<old {0.35*old+0.65*clamped} else {0.80*old+0.20*clamped});
        } else {
            *slot=Some(ms);
        }
        true
    }
    fn note_draft(&mut self,chats:usize,ms:f64) {
        if let (Some(slot),Some(n))=(self.draft_ms.get_mut(chats),self.draft_n.get_mut(chats)) {
            Self::ema(slot,n,ms);
            self.last_draft_ms=slot.unwrap_or(ms);
        } else {
            self.last_draft_ms=ms;
        }
    }
    fn note_push(&mut self,ms:f64) {
        self.push_ms=Some(self.push_ms.map_or(ms,|old|if ms<old {0.35*old+0.65*ms} else {0.75*old+0.25*ms}));
    }
    /// Smooth MoE-saturated concave prior for `total_rows` in a `chats`-chat cohort before warm sample 2 arrives.
    fn prior_at(plain:f64,chats:usize,total_rows:usize)->f64 {
        if total_rows<=chats {return plain;}
        let d_avg=(total_rows-chats) as f64/chats.max(1) as f64;
        let sat=(1.0/(chats as f64).powf(0.45)).min(0.85);
        plain*(1.0+sat*0.35*d_avg.powf(0.65))
    }
    /// Estimated warm step time (ms) for `total_rows` rows in this `chats`-chat cohort.
    fn step_at(&self,chats:usize,ids:&[u64],total_rows:usize)->f64 {
        let state=self.state_ref(chats,ids);
        let plain=state.and_then(|s|s.plain_ms).unwrap_or(30.0+2.5*chats as f64);
        if total_rows<=chats {return plain;}
        let prior=Self::prior_at(plain,chats,total_rows);
        let Some(state)=state else {return prior;};
        if let Some(ms)=state.verify_ms.get(total_rows).copied().flatten() {return ms;}
        let below=(chats+1..total_rows).rev().find_map(|r|state.verify_ms.get(r).copied().flatten().map(|ms|(r,ms)));
        let above=(total_rows+1..state.verify_ms.len()).find_map(|r|state.verify_ms.get(r).copied().flatten().map(|ms|(r,ms)));
        match (below,above) {
            (Some((lo,a)),Some((hi,b)))=>a+(b-a)*(total_rows-lo) as f64/(hi-lo) as f64,
            (Some((lo,ms)),None)=>{
                let p_lo=(Self::prior_at(plain,chats,lo)-plain).max(1.0e-6);
                let ratio=(prior-plain)/p_lo;
                (plain+(ms-plain).max(0.0)*ratio).max(ms+0.25*(total_rows-lo) as f64)
            }
            (None,Some((hi,ms)))=>{
                let p_hi=(Self::prior_at(plain,chats,hi)-plain).max(1.0e-6);
                let ratio=(prior-plain)/p_hi;
                (plain+(ms-plain).max(0.0)*ratio).max(plain*1.02)
            }
            (None,None)=>prior,
        }
    }
    fn target_rate(&self,chats:usize,ids:&[u64])->f64 {
        let state=self.state_ref(chats,ids);
        let plain=state.and_then(|s|s.plain_ms).unwrap_or(30.0+2.5*chats as f64);
        let plain_rate=chats as f64/plain.max(1.0);
        if let Some(s)=state && s.rounds>=4 && s.ms>0.0 {
            (s.tokens/s.ms).max(plain_rate)
        } else {plain_rate}
    }
    /// Whether this `chats`-chat cohort should draft this tick (false on tick 1 of a new cohort to measure its
    /// exact plain cohort speed, or while paused after sustained net loss vs plain).
    fn should_draft(&mut self,chats:usize,ids:&[u64])->bool {
        let Some(state)=self.state_mut(chats,ids) else {return true;};
        if state.plain_ms.is_none() {return false;}
        state.probing=false;
        if state.paused {
            state.plain_run+=1;
            if state.plain_run<state.gap {return false;}
            state.plain_run=0;state.probing=true;
        }
        true
    }
    /// Optimal verify depth selection:
    /// - `C2` (`chats <= 2`, `M <= 16`): exact per-chat greedy matroid selection over monotonically non-increasing
    ///   survival gains.
    /// - `C3..=C32` (`chats >= 3`): uniform cohort depth `d` chosen from a compact 3-4 entry ladder (`[1, 2, 3]`
    ///   when `max_d <= 3`, else `[1, 3, 5, 7]` + `max_d`), giving every chat identical `span.count = d + 1` so
    ///   CUDA Graphs and the device vector pool stay ~100% warm with zero per-step graph churn or allocator stalls.
    pub(super) fn select_depths(&self,chats:usize,ids:&[u64],row_cap:usize,gains:&[Vec<f64>])->Vec<usize> {
        let n=gains.len();
        let mut keep=vec![0usize;n];
        let max_extra=row_cap.saturating_sub(chats);
        if max_extra==0 || n==0 {return keep;}
        let rate=self.target_rate(chats,ids);
        let push=self.push_ms.unwrap_or(0.5);
        let base_val=chats as f64-rate*self.step_at(chats,ids,chats);
        if chats>=3 && n==chats {
            let max_d=gains.iter().map(Vec::len).min().unwrap_or(0).min(max_extra/chats);
            if max_d==0 {return keep;}
            let mut ladder:Vec<usize>=if chats>8 {
                if max_d<=1 {vec![1]} else {vec![1,max_d]}
            } else if max_d<=3 {(1..=max_d).collect()} else {
                let mut v:Vec<usize>=[1,3,5,7].into_iter().filter(|&d|d<=max_d).collect();
                if v.last().copied()!=Some(max_d) {v.push(max_d);}
                v
            };
            ladder.dedup();
            let plain=self.step_at(chats,ids,chats).max(1.0);
            // `draft` is already sunk when `select_depths` runs; evaluate post-draft Dinkelbach renewal value
            // `(chats + gain_sum) - rate * (verify_ms + push)` against dropping to plain (`chats - rate * plain`),
            // where `rate = target_rate` already reflects full round time (including draft) across observed rounds.
            let (mut best_d,mut best_val)=(0usize,chats as f64-rate*plain);
            for d in ladder {
                let gain_sum:f64=gains.iter().map(|g|g[..d].iter().sum::<f64>()).sum();
                let v_ms=(self.step_at(chats,ids,chats*(d+1))+push).max(1.0);
                let val=(chats as f64+gain_sum)-rate*v_ms;
                if val>best_val {best_d=d;best_val=val;}
            }
            keep.fill(best_d);
            return keep;
        }
        let mut cands:Vec<(usize,usize,f64)>=Vec::new();
        for (i,g) in gains.iter().enumerate() {
            for (s,&gain) in g.iter().enumerate() {cands.push((i,s,gain));}
        }
        cands.sort_by(|a,b|b.2.total_cmp(&a.2).then_with(||a.1.cmp(&b.1)).then_with(||a.0.cmp(&b.0)));
        let (mut best_k,mut best_val)=(0usize,base_val);
        let mut gain_sum=0.0f64;
        for (idx,&(_,_,gain)) in cands.iter().take(max_extra).enumerate() {
            gain_sum+=gain;
            let k=idx+1;
            let val=(chats as f64+gain_sum)-rate*(self.step_at(chats,ids,chats+k)+push);
            if val>best_val {best_k=k;best_val=val;}
        }
        for &(i,_,_) in cands.iter().take(best_k) {keep[i]+=1;}
        keep
    }
    fn start_open(&mut self,chats:usize,ids:Vec<u64>,round_ms:f64,pending_chats:usize,auto_kept:usize)->u64 {
        let id=self.next_round_id;
        self.next_round_id=self.next_round_id.wrapping_add(1).max(1);
        if pending_chats==0 {
            self.observe_round(chats,&ids,auto_kept,round_ms);
            self.open=None;
        } else {
            self.open=Some(OpenRound {id,chats,ids,round_ms,remaining:pending_chats,kept:auto_kept});
        }
        id
    }
    pub(super) fn finish_chat(&mut self,round_id:u64,keep:usize) {
        let done=if let Some(open)=self.open.as_mut().filter(|o|o.id==round_id) {
            open.kept+=keep;
            open.remaining=open.remaining.saturating_sub(1);
            (open.remaining==0).then(||(open.chats,std::mem::take(&mut open.ids),open.kept,open.round_ms))
        } else {None};
        if let Some((chats,ids,kept,round_ms))=done {
            self.open=None;
            self.observe_round(chats,&ids,kept,round_ms);
        }
    }
    fn observe_round(&mut self,chats:usize,ids:&[u64],kept:usize,round_ms:f64) {
        let Some(state)=self.state_mut(chats,ids) else {return;};
        let plain_rate=state.plain_ms.map_or(0.0,|ms|chats as f64/ms.max(1.0));
        let fade=if state.rounds<8 {1.0} else {0.95};
        state.tokens=state.tokens*fade+kept as f64;
        state.ms=state.ms*fade+round_ms.max(1.0);
        state.rounds=state.rounds.saturating_add(1);
        if std::mem::take(&mut state.probing) {
            if kept as f64/round_ms.max(1.0)>=plain_rate*0.97 {
                state.paused=false;state.gap=COHORT_PROBE_FIRST;
                state.tokens=kept as f64*4.0;state.ms=round_ms.max(1.0)*4.0;state.rounds=4;
            } else {
                state.gap=(state.gap*2).min(COHORT_PROBE_MOST);
            }
        } else if state.rounds>=COHORT_FALLBACK_MIN_ROUNDS && state.ms>0.0
            && (state.tokens/state.ms)<plain_rate*0.95 {
            state.paused=true;state.plain_run=0;state.gap=COHORT_PROBE_FIRST;
        }
    }
}

/// Flush all queued `lag` + `deferred_push` taps across a slice of contexts in ONE batched `model.push_batch` call.
fn flush_batch<C:Compute,W:Weights<C>>(c:&C,model:&dyn Model<C,W>,ctxs:&mut [&mut Context<C::Value>])->Result<bool> {
    let width=model.config().hidden_size;
    let mut queued:Vec<(usize,Vec<C::Value>,usize)>=Vec::new();
    for (idx,ctx) in ctxs.iter_mut().enumerate() {
        if let Some((taps,rows))=ctx.take_queued(c,width)? {
            queued.push((idx,taps,rows));
        }
    }
    if queued.is_empty() {return Ok(false);}
    // Pair mutable `&mut DraftChat` borrows with their extracted tap buffers.
    let mut chat_refs:Vec<Option<&mut super::super::draft::DraftChat<C::Value>>>=ctxs.iter_mut().map(|ctx|Some(&mut ctx.chat)).collect();
    let mut items=Vec::with_capacity(queued.len());
    for (idx,taps,rows) in &queued {
        let chat=chat_refs[*idx].take().ok_or("loop text service: duplicate chat in push_batch")?;
        items.push((chat,taps.as_slice(),*rows));
    }
    model.push_batch(c,&mut items)?;
    Ok(true)
}

impl<'a,C:Compute,W:Weights<C>> Service<'a,C,W> {
    pub(super) fn verify_row_cap(&self)->usize {
        if self.depth>0 {VERIFY_ROW_CAP.min(self.driver.limits().step_rows)} else {1}
    }
    pub(super) fn draft_chat_limit_value(&self)->usize {
        if self.depth>0 {DRAFT_CHAT_LIMIT.min(self.verify_row_cap()/2).max(1)} else {1}
    }

    /// Batched proposal for `B >= 1` drafting chats: flushes deferred taps in ONE `push_batch`, checks per-chat
    /// suffix lookup, runs ONE batched `propose_confident_batch`, and applies cohort-optimal confidence gating.
    pub(super) fn propose_batch_impl<'s>(&'s self,rows:&mut [DraftRows<'_,Chat<'s,'a,C,W>>])->std::result::Result<Vec<ModelProposal>,BatchError> {
        let err=|message:String|BatchError {request:None,message};
        if rows.is_empty() {return Ok(Vec::new());}
        if rows.len()==1 {
            use spark_engine::draft::Target;
            let r=&mut rows[0];
            let proposal=r.session.propose_model(r.history,r.next,r.limit).map_err(err)?
                .ok_or_else(||err("missing draft model".into()))?;
            return Ok(vec![proposal]);
        }
        let m=self.model.ok_or_else(||err("draft model missing".into()))?;
        let cfg=m.config();
        let b_count=rows.len();
        let started=Instant::now();
        // Reset per-round state on each chat and collect exact suffix lookup hits. For `b_count >= 3`, keep all
        // eligible chats in one constant-batch `propose_confident_batch` launch (warm draft CUDA graph!) and
        // substitute exact suffix lookup hits after uniform depth selection so every chat verifies the same count.
        let uniform_cohort=b_count>=3;
        let mut proposals:Vec<Option<Vec<u32>>>=vec![None;b_count];
        let mut lookup_hits:Vec<Option<Vec<u32>>>=vec![None;b_count];
        let mut neural_indices:Vec<usize>=Vec::new();
        let mut neural_depths:Vec<usize>=Vec::new();
        let mut any_wide=false;
        for (i,row) in rows.iter_mut().enumerate() {
            let d=row.session.draft.as_mut().ok_or_else(||err("draft context missing".into()))?;
            if let Some(from)=d.round {d.prices.overhead((started-from).as_secs_f64()*1000.0-d.spent);}
            d.spent=0.0;d.last_neural=false;d.last_probs.clear();d.last_draft_ms=0.0;
            let most=row.limit.min(d.depth);
            let (copied,_)=if most>=2 {d.lookup.propose_after(row.history,&[row.next],most,8)} else {(Vec::new(),0)};
            if !uniform_cohort && copied.len()>=2 {
                proposals[i]=Some(copied);
            } else if most>0 && d.read_to()+cfg.block_size<=cfg.max_position_embeddings {
                if copied.len()>=2 {lookup_hits[i]=Some(copied);}
                any_wide|=d.choose.is_multi_wide();
                neural_indices.push(i);
                neural_depths.push(most);
            } else if copied.len()>=2 {
                proposals[i]=Some(copied);
            } else {
                d.drops[0]+=1;
                proposals[i]=Some(Vec::new());
            }
        }
        let cohort_ids=sorted_ids(rows.iter().map(|r|r.session.id));
        let mut cohort=self.cohort.borrow_mut();
        let allow_neural=if neural_indices.is_empty() {false} else {cohort.should_draft(b_count,&cohort_ids)};
        drop(cohort);
        if allow_neural {
            // Cap shallow cohorts to SHALLOW_PROPOSE and cap uniform cohorts to the deepest uniform verify depth
            // that fits `verify_row_cap()` (e.g. `d <= 3` at `C32` instead of wasting 7 Markov steps).
            let max_depth=neural_depths.iter().copied().max().unwrap_or(0);
            let mut cap_depth=if any_wide || max_depth<=SHALLOW_PROPOSE {max_depth} else {SHALLOW_PROPOSE};
            if uniform_cohort {
                let max_uniform=(self.verify_row_cap()/b_count).saturating_sub(1).min(max_cohort_depth(b_count)).max(1);
                cap_depth=cap_depth.min(max_uniform);
            }
            for d in &mut neural_depths {*d=(*d).min(cap_depth);}
            // Pre-grow global draft KV planes BEFORE entering `compute.work` so CUDA VMM page maps never run
            // while resident launches are queued.
            for row in rows.iter_mut() {
                if let Some(d)=row.session.draft.as_mut() {
                    let need=d.read_to().saturating_add(cfg.block_size.saturating_mul(2));
                    d.chat.ensure_global_capacity(self.compute,cfg,need).map_err(err)?;
                }
            }
            // Flush deferred accepts and variable-row `push_batch` unrecorded BEFORE `propose_confident_batch` so
            // variable accepted counts never instantiate unique CUDA graphs or evict warm verify/draft graphs.
            let p0=Instant::now();
            let flushed=self.compute.work(|| {
                let paused=self.compute.rows_unrecorded(true)?;
                let run=(|| -> Result<bool> {
                    {
                        let mut slots=self.slots.borrow_mut();
                        for row in rows.iter() {
                            if let Ok(active)=row.session.active_slot()
                                && let Some(slot)=slots.get_mut(active).and_then(Option::as_mut)
                            {
                                self.flush_accept(slot)?;
                            }
                        }
                    }
                    let mut all_ctxs:Vec<&mut Context<C::Value>>=rows.iter_mut()
                        .filter_map(|r|r.session.draft.as_mut()).collect();
                    flush_batch(self.compute,m,&mut all_ctxs)
                })();
                let resumed=self.compute.rows_unrecorded(paused);
                let flushed=run?;
                resumed?;
                Ok(flushed)
            }).map_err(err)?;
            let push_ms=if flushed {p0.elapsed().as_secs_f64()*1000.0} else {0.0};
            let draft_started=Instant::now();
            let raw_results=self.compute.work(|| {
                // Compute all anchor embeddings in ONE batched lookup + norm launch before mutably borrowing `rows`.
                let anchors:Vec<u32>=neural_indices.iter().map(|&idx|rows[idx].next).collect();
                let width=self.driver.run.width();
                let embeds_all=self.driver.embeddings(&anchors)?;
                let embeds=(0..anchors.len()).map(|i|self.compute.range(&embeds_all,i*width,width)).collect::<Result<Vec<_>>>()?;
                let mut chats:Vec<Option<&mut super::super::draft::DraftChat<C::Value>>>=rows.iter_mut()
                    .map(|r|r.session.draft.as_mut().map(|d|&mut d.chat)).collect();
                let mut items=Vec::with_capacity(neural_indices.len());
                for (((&idx,&depth),&anchor),embed) in neural_indices.iter().zip(&neural_depths).zip(&anchors).zip(&embeds) {
                    let chat=chats[idx].take().ok_or("loop text service: missing draft chat")?;
                    items.push((chat,anchor,embed,depth));
                }
                m.propose_confident_batch(self.compute,self.driver.run.weights,&mut items)
            }).map_err(err)?;
            let draft_ms=draft_started.elapsed().as_secs_f64()*1000.0;
            let mut cohort=self.cohort.borrow_mut();
            if push_ms>0.0 {cohort.note_push(push_ms);}
            cohort.note_draft(b_count,draft_ms);
            // Build per-chat monotonically non-increasing survival gains for cohort-optimal selection.
            let lookup_extra:usize=proposals.iter().flatten().map(Vec::len).sum();
            let avail_cap=self.verify_row_cap().saturating_sub(lookup_extra);
            let mut gains:Vec<Vec<f64>>=Vec::with_capacity(neural_indices.len());
            for (&idx,(_,probs)) in neural_indices.iter().zip(&raw_results) {
                let d=rows[idx].session.draft.as_ref().expect("draft context checked");
                let mut survive=1.0f64;
                let mut g=Vec::with_capacity(probs.len());
                for (pos,&p) in probs.iter().enumerate() {survive*=d.choose.cal_pos(pos,p);g.push(survive);}
                if let Some(hit)=lookup_hits[idx].as_ref() {
                    let mut h_surv=1.0f64;
                    for slot in g.iter_mut().take(hit.len()) {
                        h_surv*=0.85;
                        if h_surv>*slot {*slot=h_surv;}
                    }
                }
                gains.push(g);
            }
            let keep_depths=cohort.select_depths(b_count,&cohort_ids,avail_cap,&gains);
            drop(cohort);
            for ((&idx,(mut drafted,probs)),keep_d) in neural_indices.iter().zip(raw_results).zip(keep_depths) {
                let d=rows[idx].session.draft.as_mut().expect("draft context checked");
                let used_lookup=keep_d>=2 && lookup_hits[idx].as_ref().is_some_and(|h|h.len()>=keep_d);
                if used_lookup {
                    let mut hit=lookup_hits[idx].take().expect("checked");
                    hit.truncate(keep_d);
                    drafted=hit;
                    d.last_neural=false;
                } else {
                    drafted.truncate(keep_d);
                    if drafted.is_empty() {d.drops[1]+=1;d.dropped_ms+=draft_ms;}
                    d.last_probs=probs;d.last_draft_ms=draft_ms;d.last_neural=true;
                }
                proposals[idx]=Some(drafted);
            }
        } else {
            for idx in neural_indices {
                if let Some(d)=rows[idx].session.draft.as_mut() {d.drops[0]+=1;}
                proposals[idx]=Some(Vec::new());
            }
        }
        let total_ms=started.elapsed().as_secs_f64()*1000.0;
        let mut out=Vec::with_capacity(b_count);
        for (i,row) in rows.iter_mut().enumerate() {
            let tokens=proposals[i].take().unwrap_or_default();
            let d=row.session.draft.as_mut().expect("draft context checked");
            d.spent+=total_ms;d.round=Some(started);
            let r=tokens.len()+1;
            if let Some(t)=d.tally.get_mut(r) {t[0]+=1;}
            if r==1 {d.tally[1][1]+=1;}
            out.push(ModelProposal {tokens,proposal_ms:total_ms,catchup_ms:0.0});
        }
        Ok(out)
    }

    /// Batched target forward/verify for `>= 2` drafted chats (`Prefill`, all-1-row `Decode`, or multi-row
    /// speculative `verify_many`), sharing every target weight read across all chats in one pass.
    pub(super) fn batch_drafted(&self,rows:&mut [BatchRows<'_,Chat<'_,'a,C,W>>],phase:Phase)->Result<Vec<Predictions>> {
        let Some(model)=self.model else {
            return Ok(rows.iter().map(|_|Predictions::Rejected("draft model missing".into())).collect());
        };
        let ids:Vec<u64>=model.config().target_layer_ids.iter().map(|&id|id as u64).collect();
        let decode=phase==Phase::Decode;
        let any_multi=decode && rows.iter().any(|r|r.tokens.len()>1);
        let all_greedy_verify=any_multi && rows.iter().all(|r|r.greedy && matches!(r.output,BatchOutput::All));
        if any_multi && all_greedy_verify {
            return self.batch_verify_many(rows,model,&ids);
        }
        if any_multi {
            // Non-greedy mixed decode fallback (rare): run each chat via `tapped`.
            return Ok(rows.iter_mut().map(|row| {
                let wanted=match row.output {BatchOutput::All=>Wanted::All,BatchOutput::Last=>Wanted::Last,BatchOutput::None=>Wanted::None};
                match row.session.tapped(row.tokens,wanted,!row.greedy && wanted!=Wanted::None,phase) {
                    Ok((tops,_)) if row.greedy && wanted!=Wanted::None=>Predictions::Greedy(tops),
                    Ok((_,logits))=>Predictions::Logits(logits.chunks(self.rows).map(<[f32]>::to_vec).collect()),
                    Err(message)=>Predictions::Rejected(message),
                }
            }).collect());
        }
        // Prefill or all-1-row Decode: pack into `step_rows` steps using `rows_taps_phase` + batched `push_batch`.
        let limit=self.driver.limits().step_rows;
        let (mut steps,mut room,mut ends)=(vec![Vec::new()],limit,self.chats);
        for (row,batch) in rows.iter().enumerate() {
            let tokens=batch.tokens.len();
            need(tokens>0 || matches!(batch.output,BatchOutput::None),"a row with no token asks for an answer")?;
            let mut from=0;
            while from<tokens {
                let mut to=tokens.min(from+room);
                let wanted=match batch.output {BatchOutput::All=>Wanted::All,BatchOutput::Last if to==tokens=>Wanted::Last,_=>Wanted::None};
                if wanted==Wanted::All {to=to.min(from+ends);}
                if room==0 || (wanted!=Wanted::None && ends==0) {steps.push(Vec::new());(room,ends)=(limit,self.chats);continue;}
                ends-=match wanted {Wanted::All=>to-from,Wanted::Last=>1,Wanted::None=>0};
                steps.last_mut().expect("starts with one step").push(Piece {row,from,to,wanted});
                room-=to-from;from=to;
            }
        }
        let mut answers:Vec<(Vec<u32>,Vec<Vec<f32>>)>=rows.iter().map(|_|Default::default()).collect();
        let called=rows.iter().map(|batch|batch.session.active_slot()).collect::<Result<Vec<_>>>()?;
        let width=model.config().hidden_size;
        let mut slots=self.slots.borrow_mut();
        for pieces in steps.iter().filter(|p|!p.is_empty()) {
            let started=Instant::now();
            let mut open:Vec<Option<&mut Slot<C::Value>>>=slots.iter_mut().map(Option::as_mut).collect();
            let mut taken=Vec::with_capacity(pieces.len());
            for piece in pieces {
                let slot=open.get_mut(called[piece.row]).and_then(Option::take).ok_or("loop text service: a row names a chat that is not open")?;
                let count=piece.to-piece.from;
                let need_pos=(slot.state.position as usize).saturating_add(slot.pending_accept.map_or(0,|(_,k)|k)).saturating_add(count);
                self.driver.ensure_capacity(&mut slot.state,need_pos)?;
                let ctx=rows[piece.row].session.draft.as_mut().ok_or("draft context missing")?;
                need(ctx.pending.is_empty(),"a draft window is still open")?;
                ctx.chat.ensure_global_capacity(self.compute,model.config(),ctx.read_to().saturating_add(count))?;
                taken.push(slot);
            }
            let (tops,logits,flushed)=self.compute.work(|| {
                if taken.iter().any(|s|s.pending_accept.is_some()) {
                    let paused=self.compute.rows_unrecorded(true)?;
                    let r=(|| -> Result<()> {for slot in &mut taken {self.flush_accept(slot)?;} Ok(())})();
                    let resumed=self.compute.rows_unrecorded(paused);
                    r?;resumed?;
                }
                let mut chats:Vec<ChatStep<'_,C::Value>>=taken.iter_mut().zip(pieces).map(|(slot,piece)| {
                    let batch=&rows[piece.row];
                    ChatStep {state:&mut slot.state,tokens:&batch.tokens[piece.from..piece.to],wanted:piece.wanted,logits:!batch.greedy && piece.wanted!=Wanted::None}
                }).collect();
                let (step,taps)=self.driver.rows_taps_phase(&mut chats,phase,&ids,|_,_|Ok(()))?;
                drop(chats);
                let logits=match &step.logits {Some(l)=>self.compute.read(l)?,None=>Vec::new()};
                // Slice per-piece taps; defer single-row decode taps for batched `push_batch`, push prefill immediately.
                let mut offset=0usize;
                let mut need_flush=false;
                for piece in pieces {
                    let count=piece.to-piece.from;
                    let piece_taps=taps.iter().map(|t|self.compute.range(t,offset*width,count*width)).collect::<Result<Vec<_>>>()?;
                    offset+=count;
                    let ctx=rows[piece.row].session.draft.as_mut().ok_or("draft context missing")?;
                    ctx.defer_push(piece_taps,count);
                    if !decode || count>1 || ctx.lagged+ctx.deferred_rows>=LAG_ROWS {need_flush=true;}
                }
                if need_flush {
                    let paused=self.compute.rows_unrecorded(true)?;
                    let mut ctxs:Vec<&mut Context<C::Value>>=rows.iter_mut().filter_map(|r|r.session.draft.as_mut()).collect();
                    let r=flush_batch(self.compute,model,&mut ctxs).map(|_|());
                    let resumed=self.compute.rows_unrecorded(paused);
                    r?;resumed?;
                }
                Ok((step.tops,logits,need_flush))
            })?;
            #[cfg(test)]
            self.batch_steps.set(self.batch_steps.get()+1);
            for slot in taken {slot.shared=pieces.len()>1;}
            let elapsed=started.elapsed();
            let ms=elapsed.as_secs_f64()*1000.0;
            let (mut top,mut word)=(0usize,0usize);
            let mut any_dropped_neural=false;
            for piece in pieces {
                let (batch,tokens)=(&mut rows[piece.row],piece.to-piece.from);
                let count=match piece.wanted {Wanted::All=>tokens,Wanted::Last=>1,Wanted::None=>0};
                batch.session.steps+=tokens as u64;batch.session.stepped+=elapsed;
                if let Some(ctx)=batch.session.draft.as_mut() {
                    ctx.spent+=ms;ctx.reads+=u32::from(!batch.greedy && count>0);
                    if decode && ctx.last_neural {any_dropped_neural=true;ctx.last_neural=false;}
                }
                let slice=tops.get(top..top+count).ok_or("loop text service: a step gave fewer top rows than asked")?;
                top+=count;
                if batch.greedy {answers[piece.row].0.extend_from_slice(slice);continue;}
                for _ in 0..count {
                    answers[piece.row].1.push(logits.get(word..word+self.rows).ok_or("loop text service: a step gave fewer logits than asked")?.to_vec());
                    word+=self.rows;
                }
            }
            need(top==tops.len() && word==logits.len(),"a step gave more answers than its rows asked for")?;
            if decode && pieces.len()>1 && pieces.iter().all(|p|p.to-p.from==1) {
                let ids=sorted_ids(pieces.iter().map(|p|rows[p.row].session.id));
                let mut cohort=self.cohort.borrow_mut();
                if !flushed {cohort.note_plain(pieces.len(),&ids,ms);}
                if any_dropped_neural {
                    let round_ms=cohort.last_draft_ms+ms;
                    cohort.observe_round(pieces.len(),&ids,pieces.len(),round_ms);
                }
            }
        }
        self.settle(&mut slots,&called)?;
        Ok(rows.iter().zip(answers).map(|(batch,(ids,logits))|if batch.greedy && !matches!(batch.output,BatchOutput::None) {Predictions::Greedy(ids)} else {Predictions::Logits(logits)}).collect())
    }

    /// Multi-chat speculative verify (`B >= 2` chats, at least one chat with `tokens.len() > 1`):
    /// packs into groups of at most `verify_row_cap()` rows (typically ONE group for all `B` chats) and runs
    /// ONE `Driver::verify_many` pass, committing any 1-row peer immediately and staging multi-row taps for `truncate`.
    fn batch_verify_many(&self,rows:&mut [BatchRows<'_,Chat<'_,'a,C,W>>],model:&dyn Model<C,W>,ids:&[u64])->Result<Vec<Predictions>> {
        let cap=self.verify_row_cap();
        let width=model.config().hidden_size;
        let called=rows.iter().map(|batch|batch.session.active_slot()).collect::<Result<Vec<_>>>()?;
        let mut answers:Vec<Option<Predictions>>=(0..rows.len()).map(|_|None).collect();
        // Group row indices into chunks of total tokens <= `cap`.
        let mut groups:Vec<Vec<usize>>=Vec::new();
        let mut cur:Vec<usize>=Vec::new();
        let mut cur_rows=0usize;
        for (i,row) in rows.iter().enumerate() {
            need(!row.tokens.is_empty(),"a verify row has no token")?;
            if !cur.is_empty() && cur_rows+row.tokens.len()>cap {
                groups.push(std::mem::take(&mut cur));cur_rows=0;
            }
            cur.push(i);cur_rows+=row.tokens.len();
        }
        if !cur.is_empty() {groups.push(cur);}
        let mut slots=self.slots.borrow_mut();
        for group in groups {
            if group.len()==1 {
                drop(slots);
                let idx=group[0];
                let r=&mut rows[idx];
                let (tops,_)=r.session.tapped(r.tokens,Wanted::All,false,Phase::Decode)?;
                answers[idx]=Some(Predictions::Greedy(tops));
                slots=self.slots.borrow_mut();
                continue;
            }
            let total_rows:usize=group.iter().map(|&i|rows[i].tokens.len()).sum();
            let has_multi=group.iter().any(|&i|rows[i].tokens.len()>1);
            let started=Instant::now();
            let mut open:Vec<Option<&mut Slot<C::Value>>>=slots.iter_mut().map(Option::as_mut).collect();
            let mut taken=Vec::with_capacity(group.len());
            for &idx in &group {
                let slot=open.get_mut(called[idx]).and_then(Option::take).ok_or("loop text service: a row names a chat that is not open")?;
                let count=rows[idx].tokens.len();
                let need_pos=(slot.state.position as usize).saturating_add(slot.pending_accept.map_or(0,|(_,k)|k)).saturating_add(count);
                self.driver.ensure_capacity(&mut slot.state,need_pos)?;
                let ctx=rows[idx].session.draft.as_mut().ok_or("draft context missing")?;
                need(ctx.pending.is_empty(),"a draft window is still open")?;
                ctx.chat.ensure_global_capacity(self.compute,model.config(),ctx.read_to().saturating_add(count))?;
                taken.push(slot);
            }
            let group_ids=sorted_ids(group.iter().map(|&idx|rows[idx].session.id));
            let mut pending_chats=0usize;
            let mut auto_kept=0usize;
            let tops=self.compute.work(|| {
                let any_pending=taken.iter().any(|s|s.pending_accept.is_some())
                    || rows.iter().enumerate().any(|(i,r)|group.contains(&i) && r.session.draft.as_ref().is_some_and(|ctx|ctx.lagged>0 || ctx.deferred_rows>0));
                if any_pending {
                    let paused=self.compute.rows_unrecorded(true)?;
                    let r=(|| -> Result<()> {
                        for slot in &mut taken {self.flush_accept(slot)?;}
                        let mut flush_ctxs:Vec<&mut Context<C::Value>>=Vec::new();
                        for (i,r) in rows.iter_mut().enumerate() {
                            if group.contains(&i) && let Some(ctx)=r.session.draft.as_mut() && (ctx.lagged>0 || ctx.deferred_rows>0) {
                                flush_ctxs.push(ctx);
                            }
                        }
                        if !flush_ctxs.is_empty() {flush_batch(self.compute,model,&mut flush_ctxs)?;}
                        Ok(())
                    })();
                    let resumed=self.compute.rows_unrecorded(paused);
                    r?;resumed?;
                }
                if has_multi {
                    let mut verify_chats:Vec<(&mut super::super::state::State<C::Value>,&[u32])>=taken.iter_mut().zip(&group)
                        .map(|(slot,&idx)|(&mut slot.state,rows[idx].tokens)).collect();
                    let (tops,taps)=self.driver.verify_many(&mut verify_chats,ids)?;
                    drop(verify_chats);
                    let mut offset=0usize;
                    for (slot,&idx) in taken.iter_mut().zip(&group) {
                        let count=rows[idx].tokens.len();
                        let chat_taps=taps.iter().map(|t|self.compute.range(t,offset*width,count*width)).collect::<Result<Vec<_>>>()?;
                        offset+=count;
                        let ctx=rows[idx].session.draft.as_mut().ok_or("draft context missing")?;
                        if count>1 {
                            ctx.pending=chat_taps;ctx.rows=count;
                            pending_chats+=1;
                        } else {
                            // 1-row peer riding inside a multi-chat verify pass: commit immediately so `truncate` is a no-op.
                            self.driver.accept(&mut slot.state,1,1)?;
                            ctx.defer_push(chat_taps,1);
                            auto_kept+=1;
                        }
                    }
                    Ok(tops)
                } else {
                    let mut step_chats:Vec<ChatStep<'_,C::Value>>=taken.iter_mut().zip(&group)
                        .map(|(slot,&idx)|ChatStep {state:&mut slot.state,tokens:rows[idx].tokens,wanted:Wanted::All,logits:false}).collect();
                    let (step,taps)=self.driver.rows_taps_phase(&mut step_chats,Phase::Decode,ids,|_,_|Ok(()))?;
                    drop(step_chats);
                    for (pos,&idx) in group.iter().enumerate() {
                        let chat_taps=taps.iter().map(|t|self.compute.range(t,pos*width,width)).collect::<Result<Vec<_>>>()?;
                        let ctx=rows[idx].session.draft.as_mut().ok_or("draft context missing")?;
                        ctx.defer_push(chat_taps,1);
                        auto_kept+=1;
                    }
                    Ok(step.tops)
                }
            })?;
            #[cfg(test)]
            self.batch_steps.set(self.batch_steps.get()+1);
            for slot in taken {slot.shared=true;}
            let elapsed=started.elapsed();
            let ms=elapsed.as_secs_f64()*1000.0;
            let round_id={
                let mut cohort=self.cohort.borrow_mut();
                let eff_ms=if has_multi {
                    let warm=cohort.note_verify(group.len(),&group_ids,total_rows,ms);
                    if warm {ms} else {ms.min(cohort.step_at(group.len(),&group_ids,total_rows))}
                } else {
                    cohort.note_plain(group.len(),&group_ids,ms);
                    ms
                };
                let round_ms=cohort.last_draft_ms+eff_ms;
                cohort.start_open(group.len(),group_ids,round_ms,pending_chats,auto_kept)
            };
            let mut top_off=0usize;
            for &idx in &group {
                let count=rows[idx].tokens.len();
                let batch=&mut rows[idx];
                batch.session.steps+=count as u64;batch.session.stepped+=elapsed;
                if let Some(ctx)=batch.session.draft.as_mut() {
                    ctx.spent+=ms;
                    if count>1 {ctx.batch_round=Some(round_id);} else {ctx.last_neural=false;}
                }
                let slice=tops.get(top_off..top_off+count).ok_or("loop text service: verify_many gave fewer tops than rows")?;
                top_off+=count;
                answers[idx]=Some(Predictions::Greedy(slice.to_vec()));
            }
            need(top_off==tops.len(),"verify_many gave more tops than rows")?;
        }
        self.settle(&mut slots,&called)?;
        Ok(answers.into_iter().map(|a|a.expect("every verify row answered")).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{COHORT_FALLBACK_MIN_ROUNDS,COHORT_PROBE_FIRST,CohortPricing,VERIFY_ROW_CAP};

    #[test]
    fn cohort_pricing_isolates_timings_rates_and_pauses_across_distinct_cohorts() {
        let mut p=CohortPricing::default();
        let c4_a=[1u64,2,3,4];
        assert!(!p.should_draft(4,&c4_a),"first tick measures cohort plain speed");
        p.note_plain(4,&c4_a,30.0);
        p.note_verify(4,&c4_a,16,67.0);
        for _ in 0..COHORT_FALLBACK_MIN_ROUNDS {
            assert!(p.should_draft(4,&c4_a));
            p.observe_round(4,&c4_a,4,36.0);
        }
        assert!(!p.should_draft(4,&c4_a),"sustained loss pauses cohort A");
        // A new C4 cohort (e.g. diverse prompts) starts fresh: unpaused and unaffected by cohort A's 30ms/67ms timings.
        let c4_b=[10u64,11,12,13];
        assert!(!p.should_draft(4,&c4_b),"new cohort measures its own tick-1 plain speed");
        p.note_plain(4,&c4_b,49.0);
        assert!(p.should_draft(4,&c4_b),"new cohort is never paused by an earlier cohort");
        // A C8 cohort never inherits C4's 67ms M=16 timing when C8 plain is 33ms, and a single cold capture spike
        // (`verify_n == 1`) is ignored so warm sample 2 runs and populates `verify_ms` cleanly.
        let c8=[20u64,21,22,23,24,25,26,27];
        assert!(!p.should_draft(8,&c8));
        p.note_plain(8,&c8,33.0);
        assert!(p.should_draft(8,&c8));
        assert!(!p.note_verify(8,&c8,16,95.0),"cold sample 1 ignored");
        assert!(p.step_at(8,&c8,16)<45.0,"cold sample 1 leaves prior intact");
        let gains:Vec<Vec<f64>>=(0..8).map(|_|vec![0.85]).collect();
        let keep=p.select_depths(8,&c8,VERIFY_ROW_CAP,&gains);
        assert_eq!(keep,vec![1;8],"confident C8 cohort drafts uniformly on fresh/cold prior");
        assert!(p.note_verify(8,&c8,16,44.0),"warm sample 2 accepted");
        assert!((p.step_at(8,&c8,16)-44.0).abs()<1e-9,"warm sample 2 recorded");
        // Wide C16 and C32 cohorts select deep uniform depths from the compact ladder.
        let c16:Vec<u64>=(100..116).collect();
        p.note_plain(16,&c16,70.0);
        let g16:Vec<Vec<f64>>=(0..16).map(|_|vec![0.85,0.72,0.55]).collect();
        let k16=p.select_depths(16,&c16,VERIFY_ROW_CAP,&g16);
        assert_eq!(k16,vec![3;16],"C16 drafts uniformly at d=3");
        let c32:Vec<u64>=(200..232).collect();
        p.note_plain(32,&c32,110.0);
        let g32:Vec<Vec<f64>>=(0..32).map(|_|vec![0.85,0.72,0.55]).collect();
        let k32=p.select_depths(32,&c32,VERIFY_ROW_CAP,&g32);
        assert_eq!(k32,vec![3;32],"C32 drafts uniformly at d=3");
    }

    #[test]
    fn select_depths_allocates_rows_by_survival_gain_and_handles_cold_capture_and_probe() {
        let mut p=CohortPricing::default();
        // C2 keeps per-chat greedy matroid allocation across heterogeneous chats.
        let c2=[1u64,2];
        p.note_plain(2,&c2,35.0);
        assert!(!p.note_verify(2,&c2,6,110.0),"cold sample 1 ignored");
        assert!(p.note_verify(2,&c2,6,48.0),"warm sample 2 accepted");
        assert!((p.step_at(2,&c2,6)-48.0).abs()<1e-9);
        let g2=vec![vec![0.95,0.90,0.85],vec![0.05,0.03,0.01]];
        let k2=p.select_depths(2,&c2,VERIFY_ROW_CAP,&g2);
        assert_eq!(k2,vec![3,0],"C2 allocates per chat");
        // C4+ selects uniform depth from the compact ladder to keep CUDA graphs and vector pool warm.
        let ids=[1u64,2,3,4];
        p.note_plain(4,&ids,48.0);
        let _=p.note_verify(4,&ids,16,140.0);
        let _=p.note_verify(4,&ids,16,70.0);
        let _=p.note_verify(4,&ids,8,120.0);
        let _=p.note_verify(4,&ids,8,56.0);
        let gains=vec![vec![0.95,0.90,0.85],vec![0.92,0.85,0.80],vec![0.88,0.82,0.75],vec![0.85,0.78,0.70]];
        let keep=p.select_depths(4,&ids,VERIFY_ROW_CAP,&gains);
        assert_eq!(keep,vec![3;4],"C4 selects uniform depth 3 across cohort");
        // Sustained negative return pauses, then a profitable probe resumes drafting.
        for _ in 0..COHORT_FALLBACK_MIN_ROUNDS {
            let _=p.should_draft(4,&ids);
            p.observe_round(4,&ids,4,56.0);
        }
        for _ in 1..COHORT_PROBE_FIRST {assert!(!p.should_draft(4,&ids));}
        assert!(p.should_draft(4,&ids),"probe fires after COHORT_PROBE_FIRST plain steps");
        p.observe_round(4,&ids,12,72.0);
        assert!(p.should_draft(4,&ids),"profitable probe resumes drafting");
    }
}
