//! The scheduler owns the draft window. A multi-row verify stays provisional until
//! the scheduler supplies the committed position through Target::truncate.
use super::*;
use crate::loop_decoder::{draft::{Drafter,DraftChat},operands::weights::ResidentProduct};
use spark_engine::draft::{ModelProposal,Target};

pub trait Model<C:Compute,W:Weights<C>> {
    fn config(&self)->&engine_model::mtp_draft::MtpConfig;
    /// Maximum neural draft depth supported by `propose_confident` (defaults to `config().gamma()`;
    /// GPU-packed DFlash supports wide 3-tier block-causal proposals up to `MAX_POS = 15`).
    fn wide_depth(&self)->usize {self.config().gamma()}
    fn chat(&self)->DraftChat<C::Value>;
    fn push(&self,c:&C,chat:&mut DraftChat<C::Value>,taps:&[C::Value],rows:usize)->Result<()>;
    fn propose(&self,c:&C,w:&W,chat:&DraftChat<C::Value>,anchor:u32,embed:&C::Value,depth:usize)->Result<Vec<u32>>;
    /// `propose` with the top-1 softmax probability of each draft token (1.0 each for a model that gives none).
    #[allow(clippy::too_many_arguments)]
    fn propose_confident(&self,c:&C,w:&W,chat:&mut DraftChat<C::Value>,anchor:u32,embed:&C::Value,depth:usize,_abort_below:f32)->Result<(Vec<u32>,Vec<f32>)> {
        let tokens=self.propose(c,w,chat,anchor,embed,depth)?;
        let probs=vec![1.0;tokens.len()];
        Ok((tokens,probs))
    }
    /// Batched `push` across multiple chats sharing draft projection weights.
    fn push_batch(&self,c:&C,items:&mut [(&mut DraftChat<C::Value>,&[C::Value],usize)])->Result<()> {
        for (chat,taps,rows) in items {
            if *rows>0 {self.push(c,chat,taps,*rows)?;}
        }
        Ok(())
    }
    /// Batched `propose_confident` across multiple chats sharing draft weights.
    fn propose_confident_batch(&self,c:&C,w:&W,items:&mut [(&mut DraftChat<C::Value>,u32,&C::Value,usize)])->Result<Vec<(Vec<u32>,Vec<f32>)>> {
        items.iter_mut().map(|(chat,anchor,embed,depth)|self.propose_confident(c,w,chat,*anchor,embed,*depth,0.0)).collect()
    }
}
impl<C:ResidentProduct<V>+super::super::draft::Lanes,W:Weights<C>,V> Model<C,W> for Drafter<'_,V,C::Value> {
    fn config(&self)->&engine_model::mtp_draft::MtpConfig {self.config()}
    fn wide_depth(&self)->usize {self.wide_depth()}
    fn chat(&self)->DraftChat<C::Value> {self.chat()}
    fn push(&self,c:&C,chat:&mut DraftChat<C::Value>,taps:&[C::Value],rows:usize)->Result<()> {self.push_values(c,chat,taps,rows)}
    fn propose(&self,c:&C,w:&W,chat:&DraftChat<C::Value>,anchor:u32,embed:&C::Value,depth:usize)->Result<Vec<u32>> {
        self.propose_value(c,w,chat,anchor,embed,depth)
    }
    fn propose_confident(&self,c:&C,w:&W,chat:&mut DraftChat<C::Value>,anchor:u32,embed:&C::Value,depth:usize,abort_below:f32)->Result<(Vec<u32>,Vec<f32>)> {
        self.propose_confident(c,w,chat,anchor,embed,depth,abort_below)
    }
    fn push_batch(&self,c:&C,items:&mut [(&mut DraftChat<C::Value>,&[C::Value],usize)])->Result<()> {
        self.push_values_batch(c,items)
    }
    fn propose_confident_batch(&self,c:&C,w:&W,items:&mut [(&mut DraftChat<C::Value>,u32,&C::Value,usize)])->Result<Vec<(Vec<u32>,Vec<f32>)>> {
        self.propose_confident_batch(c,w,items)
    }
}
/// The draft window of a chat: the draft context, the resident depth of the request, the online logistic acceptance
/// estimator and smooth 2-regime round prices of the chat, a multi-length exact suffix lookup (`8/6/4/3/2/1`), the
/// taps of an open check step and its rows, and the taps of the plain steps that the draft context has not read yet.
pub(super) struct Context<V> {
    pub chat:DraftChat<V>,pub depth:usize,pub(super) choose:Chooser,pub pending:Vec<V>,pub rows:usize,pub lag:Vec<V>,pub lagged:usize,
    pub(super) deferred_push:Vec<(Vec<V>,usize)>,pub(super) deferred_rows:usize,pub(super) batch_round:Option<u64>,
    pub(super) prices:Prices,pub(super) lookup:spark_engine::flash_next_lookup::SuffixLookup,pub(super) tally:[[u32;2];17],pub(super) reads:u32,
    pub(super) round:Option<Instant>,pub(super) spent:f64,pub(super) last_probs:Vec<f32>,pub(super) last_draft_ms:f64,pub(super) last_neural:bool,
    last_neural_len:usize,last_ext_bucket:Option<usize>,last_direct_lookup:bool,
    pub(super) drops:[u32;3],pub(super) dropped_ms:f64,ext_stats:[u32;4],
}
impl<V> Drop for Context<V> {
    fn drop(&mut self) {
        let rows=|k:usize|self.tally.iter().enumerate().filter(|(_,t)|t[0]>0).map(|(r,t)|format!("{r}:{}",t[k])).collect::<Vec<_>>().join(",");
        let max_r=self.tally.iter().enumerate().rev().find(|(_,t)|t[0]>0).map_or(self.depth+1,|(r,_)|r.max(self.depth+1));
        let seen=(1..=max_r).map(|rows|self.prices.round(rows).map_or("-".into(),|ms|format!("{ms:.1}"))).collect::<Vec<_>>().join(",");
        let ms=|v:Option<f64>|v.map_or("-".into(),|ms|format!("{ms:.2}"));
        eprintln!("mtp_rounds depth={} rounds={} kept={} logit_reads={} round_ms={seen} plain={} dropped={} early={} dropped_ms={:.0} step1_ms={} overhead_ms={} push_ms={} omega={:.3} b0={:.3} dir={}/{} ext={}/{}",
            self.depth,rows(0),rows(1),self.reads,self.drops[0],self.drops[1],self.drops[2],self.dropped_ms,
            ms(self.prices.step1),ms(self.prices.overhead),ms(self.prices.push),
            self.choose.omega,self.choose.beta[0],self.ext_stats[0],self.ext_stats[1],self.ext_stats[2],self.ext_stats[3]);
    }
}
/// The most plain steps whose taps wait for the draft context. A plain step then pushes nothing: the waiting rows go
/// into the draft context in one push when a draft needs them, or when this many wait. A push of many rows costs
/// about what a push of one row costs, so plain steps of a chat whose drafts do not pay cost about a plain chat's.
pub(super) const LAG_ROWS:usize=32;
impl<V> Context<V> {
    pub(super) fn new(chat:DraftChat<V>,depth:usize)->Self {
        Self {chat,depth,choose:Chooser::new(),pending:Vec::new(),rows:0,lag:Vec::new(),lagged:0,
            deferred_push:Vec::new(),deferred_rows:0,batch_round:None,
            prices:Prices::default(),lookup:Default::default(),tally:[[0;2];17],reads:0,round:None,spent:0.0,
            last_probs:Vec::new(),last_draft_ms:0.0,last_neural:false,last_neural_len:0,last_ext_bucket:None,
            last_direct_lookup:false,drops:[0;3],dropped_ms:0.0,ext_stats:[0;4]}
    }
    /// The position of the draft context with the rows that wait in `lag` and `deferred_push`.
    pub(super) fn read_to(&self)->usize {self.chat.len()+self.lagged+self.deferred_rows}
    /// Copy the taps of one plain row (`[width]` each) after the waiting rows; the draft context reads them later.
    fn wait<C:Compute<Value=V>,W:Weights<C>>(&mut self,c:&C,model:&dyn Model<C,W>,taps:&[V])->Result<()> {
        if self.deferred_rows>0 {self.catch_up(c,model)?;}
        let width=model.config().hidden_size;
        need(taps.iter().all(|tap|c.words(tap)==width) && self.lagged<LAG_ROWS,"a waiting draft row differs from the width")?;
        if self.lag.len()!=taps.len() {self.lag=taps.iter().map(|_|c.zeros(LAG_ROWS*width)).collect::<Result<Vec<_>>>()?;}
        for (rows,tap) in self.lag.iter_mut().zip(taps) {c.state_fill(rows,self.lagged*width,tap)?;}
        self.lagged+=1;
        if self.lagged==LAG_ROWS {self.catch_up(c,model)?;}
        Ok(())
    }
    /// Queue a batch-produced tap slice for batched `push_batch` on the next cohort proposal.
    pub(super) fn defer_push(&mut self,taps:Vec<V>,rows:usize) {
        if rows==0 {return;}
        self.deferred_push.push((taps,rows));
        self.deferred_rows+=rows;
    }
    /// Extract all queued `lag` + `deferred_push` rows in chronological order as one `(taps, rows)` slice.
    pub(super) fn take_queued<C:Compute<Value=V>>(&mut self,c:&C,width:usize)->Result<Option<(Vec<V>,usize)>> {
        if self.lagged==0 && self.deferred_rows==0 {return Ok(None);}
        let mut chunks:Vec<(Vec<V>,usize)>=Vec::new();
        if self.lagged>0 {
            let rows=self.lagged;
            let parts=self.lag.iter().map(|tap|c.range(tap,0,rows*width)).collect::<Result<Vec<_>>>()?;
            self.lagged=0;
            chunks.push((parts,rows));
        }
        if !self.deferred_push.is_empty() {
            for (taps,rows) in self.deferred_push.drain(..) {
                let sliced=if taps.first().is_some_and(|v|c.words(v)!=rows*width) {
                    taps.iter().map(|v|c.range(v,0,rows*width)).collect::<Result<Vec<_>>>()?
                } else {taps};
                chunks.push((sliced,rows));
            }
            self.deferred_rows=0;
        }
        if chunks.len()==1 {return Ok(chunks.pop());}
        let total:usize=chunks.iter().map(|(_,r)|*r).sum();
        let n_taps=chunks[0].0.len();
        let mut joined=Vec::with_capacity(n_taps);
        for t in 0..n_taps {
            let refs=chunks.iter().map(|(v,_)|&v[t]).collect::<Vec<_>>();
            joined.push(c.join(&refs)?);
        }
        Ok(Some((joined,total)))
    }
    /// Push the waiting rows into the draft context, one push for all.
    pub(super) fn catch_up<C:Compute<Value=V>,W:Weights<C>>(&mut self,c:&C,model:&dyn Model<C,W>)->Result<()> {
        if let Some((parts,rows))=self.take_queued(c,model.config().hidden_size)? {
            model.push(c,&mut self.chat,&parts,rows)?;
        }
        Ok(())
    }
}

/// Maximum verified draft tokens per round (`M = 1..=16` total rows).
const MAX_POS:usize=15;
/// Maximum hybrid suffix-extension tokens appended past a neural block (`d=7 -> d=15`).
const EXT_OFFS:usize=8;
/// Six exact suffix match lengths tracked separately (`8, 6, 4, 3, 2, 1`).
const EXT_BUCKETS:usize=6;
/// Model-agnostic initial logistic calibration (`p_hat_i = sigmoid(omega * logit(q_max_i) + beta_i)`).
const OMEGA_INIT:f64=0.60;
const BETA_INIT:f64=-0.38;
/// MAP ridge regularization and damping prior for online arrowhead IRLS updates.
/// Strong prior on shared slope `omega` prevents small-batch overfitting while per-position `beta_i` tracks workload shifts.
const IRLS_LAMBDA_OMEGA:f64=24.0;
const IRLS_LAMBDA_BETA:f64=1.5;
const IRLS_DAMP_PRIOR:f64=4.0;
const IRLS_EARLY_FIT:u32=8;
const IRLS_FIT_EVERY:u32=16;
/// A chat paused on plain steps probes after this many of them, backing off up to `PROBE_MOST`.
const PROBE_FIRST:usize=8;
const PROBE_MOST:usize=64;
/// Floor on adaptive neural propose depth during sustained short-chain stretches.
const MIN_PROPOSE:usize=3;
/// Shallow multi-chat propose cap while recent cohort rounds were short.
pub(super) const SHALLOW_PROPOSE:usize=MIN_PROPOSE;
/// Base single-block neural propose depth (`block_size - 1`); chats open past this (`11..=15`)
/// only on ceiling hits when wide neural acceptance pays its own draft+verify cost.
const BASE_PROPOSE:usize=7;
/// Consecutive rounds with `rows <= BASE_PROPOSE + 1` before stepping back down from wide neural drafting.
const WIDE_COOLDOWN:u32=2;
/// Minimum consecutive base-ceiling hits (`keep >= 8`) to open wide neural drafting from `7 -> 11` on a fresh chat.
const WIDE_OPEN_STREAK:u32=2;
/// Decaying EWMA threshold of deep rounds (`keep >= 10`) that identifies a sustained deep-acceptance regime
/// (holds `d=15` across single hard tokens and re-opens directly to `15`, while decaying cleanly off on Code).
const DEEP_REGIME_EWMA:f64=0.25;
/// Drafted rounds run before comparing kept tokens/ms against plain steps for pause fallback.
const FALLBACK_MIN_ROUNDS:u32=24;

fn logit(p:f32)->f64 {
    let q=f64::from(p).clamp(1.0e-4,1.0-1.0e-4);
    (q/(1.0-q)).ln()
}
fn sigmoid(z:f64)->f64 {
    if z>=0.0 {1.0/(1.0+(-z).exp())} else {let e=z.exp();e/(1.0+e)}
}
fn ext_bucket(match_len:usize)->Option<usize> {
    match match_len {8=>Some(0),6=>Some(1),4=>Some(2),3=>Some(3),2=>Some(4),1=>Some(5),_=>None}
}

/// Online per-position logistic acceptance estimator (`omega`, `beta[0..15]`) fitted via exact `O(K)`
/// MAP-regularized damped arrowhead IRLS, paired with Dinkelbach optimal prefix truncation, deadlock-free
/// adaptive propose depth, safe chain-aware early abort, and online-calibrated direct + hybrid suffix lookup up to `M=16`.
/// A choice changes only how many draft rows are verified, never any output token.
pub(super) struct Chooser {
    omega:f64,beta:[f64;MAX_POS],seen_pos:[u32;MAX_POS],
    g0:f64,g:[f64;MAX_POS],h00:f64,h0:[f64;MAX_POS],h:[f64;MAX_POS],n0:f64,n:[f64;MAX_POS],
    fit_rounds:u32,total_fits:u32,
    neural_tokens:f64,neural_ms:f64,neural_rounds:u32,
    cond_chain:f64,propose_d:usize,short_streak:u32,wide_miss:u32,wide_streak:u32,deep_ewma:f64,consec_drops:u32,
    dir_first_h:f64,dir_first_t:f64,dir_cont_h:f64,dir_cont_t:f64,dir_rounds:u32,dir_skipped:u32,
    ext_hits:[[f64;EXT_OFFS];EXT_BUCKETS],ext_trials:[[f64;EXT_OFFS];EXT_BUCKETS],
    multi_wide:bool,multi_low_streak:u32,
    paused:bool,plain_run:usize,gap:usize,probing:bool,
}
impl Chooser {
    fn new()->Self {
        let ext_priors=[0.76,0.72,0.66,0.64,0.62,0.52];
        let mut ext_hits=[[0.0;EXT_OFFS];EXT_BUCKETS];
        let ext_trials=[[2.0;EXT_OFFS];EXT_BUCKETS];
        for (b,&r) in ext_priors.iter().enumerate() {ext_hits[b]=[r*2.0;EXT_OFFS];}
        let beta=[BETA_INIT;MAX_POS];
        Self {
            omega:OMEGA_INIT,beta,seen_pos:[0;MAX_POS],
            g0:0.0,g:[0.0;MAX_POS],h00:0.0,h0:[0.0;MAX_POS],h:[0.0;MAX_POS],n0:0.0,n:[0.0;MAX_POS],
            fit_rounds:0,total_fits:0,
            neural_tokens:0.0,neural_ms:0.0,neural_rounds:0,
            cond_chain:2.5,propose_d:BASE_PROPOSE,short_streak:0,wide_miss:0,wide_streak:0,deep_ewma:0.0,consec_drops:0,
            dir_first_h:2.5,dir_first_t:3.0,dir_cont_h:2.7,dir_cont_t:3.0,dir_rounds:0,dir_skipped:0,
            ext_hits,ext_trials,
            multi_wide:false,multi_low_streak:0,
            paused:false,plain_run:0,gap:PROBE_FIRST,probing:false,
        }
    }
    /// Whether this chat has shown sustained multi-token acceptance in a multi-chat cohort.
    pub(super) fn is_multi_wide(&self)->bool {self.multi_wide}
    /// Update online IRLS calibration and multi-chat widen flag after a multi-chat verify round without mixing
    /// cohort timing into single-chat `neural_ms` / `paused`.
    pub(super) fn observe_multi_chat(&mut self,probs:&[f32],rows:usize,keep:usize) {
        if rows>1 {
            let accepted=keep.saturating_sub(1);
            let checked=(rows-1).min(probs.len()).min(MAX_POS);
            for (s,&p) in probs.iter().enumerate().take(checked) {
                if s>accepted {break;}
                let x=logit(p);
                let p_hat=self.cal_pos(s,p);
                let y=f64::from(u8::from(s<accepted));
                let err=y-p_hat;
                let w=(p_hat*(1.0-p_hat)).max(1.0e-6);
                self.g0+=err*x;self.g[s]+=err;
                self.h00+=w*x*x;self.h0[s]+=w*x;self.h[s]+=w;
                self.n0+=1.0;self.n[s]+=1.0;
            }
            self.fit_rounds=self.fit_rounds.saturating_add(1);
            let target_fit=if self.total_fits<3 {IRLS_EARLY_FIT} else {IRLS_FIT_EVERY};
            if self.fit_rounds>=target_fit {self.refit_irls();}
        }
        let hit_end=rows>SHALLOW_PROPOSE && (keep==rows || (keep>=SHALLOW_PROPOSE && probs.get(rows-2).is_some_and(|&p|p>=0.78)));
        if hit_end {self.multi_wide=true;self.multi_low_streak=0;}
        else if keep<=3 {self.multi_low_streak+=1;if self.multi_low_streak>=4 {self.multi_wide=false;}}
        else {self.multi_low_streak=0;}
    }
    /// Calibrated acceptance probability of neural draft position `pos` with top-1 probability `p`.
    pub(super) fn cal_pos(&self,pos:usize,p:f32)->f64 {
        let i=pos.min(MAX_POS-1);
        sigmoid(self.omega*logit(p)+self.beta[i]).clamp(0.02,0.995)
    }
    /// Calibrated conditional acceptance probability of direct 8-token suffix offset `off`.
    fn dir_cal(&self,off:usize)->f64 {
        if off==0 {(self.dir_first_h/self.dir_first_t).clamp(0.02,0.995)}
        else {(self.dir_cont_h/self.dir_cont_t).clamp(0.02,0.995)}
    }
    /// Calibrated conditional acceptance probability of hybrid suffix extension offset `off`,
    /// with a small optimistic UCB bonus while the bucket has fewer than ~8 samples.
    fn ext_cal(&self,bucket:usize,off:usize,explore:bool)->f64 {
        let i=off.min(EXT_OFFS-1);
        let t=self.ext_trials[bucket][i];
        let mean=self.ext_hits[bucket][i]/t;
        let bonus=if explore && t<8.0 {0.22*(1.0/(t+1.0)).sqrt()} else {0.0};
        (mean+bonus).clamp(0.02,0.98)
    }
    /// Expected kept tokens of verifying `rows` rows given per-position acceptance probabilities `cal`.
    fn expected_from(cal:&[f64],rows:usize)->f64 {
        let (mut sum,mut survive)=(1.0,1.0);
        for &p in &cal[..rows.saturating_sub(1).min(cal.len())] {survive*=p;sum+=survive;}
        sum
    }
    /// Target throughput rate in tokens/ms: opportunity cost of every extra verify millisecond.
    fn target_rate(&self,prices:&Prices)->f64 {
        let plain=prices.step_at(1).map_or(1.0/35.0,|ms|1.0/ms.max(1.0));
        if self.neural_rounds>=4 && self.neural_ms>0.0 {(self.neural_tokens/self.neural_ms).max(plain)} else {plain}
    }
    /// Chain-aware early-abort threshold on `q_max[0]`: only aborts after step 0 on short-chain chats
    /// (`cond_chain < 2.5`) when `p_0` is so low that even a conditional chain cannot beat plain decode.
    /// Wide high-acceptance chats (`cond_chain >= 2.5`) keep `0.0` (zero mid-draft syncs, full chain check).
    fn abort_below(&self,prices:&Prices)->f32 {
        if self.neural_rounds<8 || self.consec_drops>=12 || self.cond_chain>=2.5 {return 0.0;}
        let Some((one,two))=prices.step_at(1).zip(prices.step_at(2)) else {return 0.0};
        let push=prices.push.unwrap_or(0.35);
        let need_p0=(self.target_rate(prices)*(two+push-one))/(0.35*self.cond_chain.max(1.0)+0.65);
        let target_p=need_p0.clamp(0.0,0.40);
        if target_p<=0.05 {return 0.0;}
        let z_target=(target_p/(1.0-target_p)).ln();
        let q=sigmoid((z_target-self.beta[0])/self.omega.max(0.15));
        (q as f32).clamp(0.0,0.45)
    }
    /// Exact `O(K)` MAP-regularized Schur-complement arrowhead IRLS step on `(H + P) * delta = g`.
    fn refit_irls(&mut self) {
        let mut denom=self.h00+IRLS_LAMBDA_OMEGA;
        let mut rhs=self.g0-IRLS_LAMBDA_OMEGA*(self.omega-OMEGA_INIT);
        for i in 0..MAX_POS {
            let di=self.h[i]+IRLS_LAMBDA_BETA;
            let gi=self.g[i]-IRLS_LAMBDA_BETA*(self.beta[i]-BETA_INIT)*0.25;
            denom-=(self.h0[i]*self.h0[i])/di;
            rhs-=(self.h0[i]*gi)/di;
        }
        let d0=if denom>1.0e-9 {rhs/denom} else {0.0};
        let damp0=self.n0/(self.n0+IRLS_DAMP_PRIOR);
        self.omega=(self.omega+(d0*damp0).clamp(-0.20,0.20)).clamp(0.35,1.02);
        for i in 0..MAX_POS {
            if self.n[i]>0.0 {
                let di=self.h[i]+IRLS_LAMBDA_BETA;
                let gi=self.g[i]-IRLS_LAMBDA_BETA*(self.beta[i]-BETA_INIT)*0.25;
                let step=(gi-self.h0[i]*d0)/di;
                let damp=self.n[i]/(self.n[i]+IRLS_DAMP_PRIOR);
                self.beta[i]=(self.beta[i]+(step*damp).clamp(-0.5,0.5)).clamp(-4.0,4.0);
                self.seen_pos[i]=self.seen_pos[i].saturating_add(1);
            } else if self.seen_pos[i]==0 && i>0 {
                // Unobserved deeper positions inherit the nearest observed position's learned workload shift.
                self.beta[i]=self.beta[i-1];
            }
        }
        self.g0=0.0;self.g=[0.0;MAX_POS];self.h00=0.0;self.h0=[0.0;MAX_POS];self.h=[0.0;MAX_POS];
        self.n0=0.0;self.n=[0.0;MAX_POS];self.fit_rounds=0;self.total_fits=self.total_fits.saturating_add(1);
    }
    /// Decide whether a pre-draft 8-token suffix match (`available` tokens) beats the neural drafter;
    /// returns `0` to skip direct lookup and run the neural drafter, or `2..=available+1` rows to verify.
    fn direct_lookup_rows(&mut self,prices:&Prices,available:usize)->usize {
        if available<2 || !prices.plain_known() {return 0;}
        let n=available.min(MAX_POS);
        let mut cal=[0.0;MAX_POS];
        for (i,slot) in cal[..n].iter_mut().enumerate() {*slot=self.dir_cal(i);}
        let rate=self.target_rate(prices);
        let push=prices.push.unwrap_or(0.35);
        let overhead=prices.overhead.unwrap_or(0.05);
        let (mut best_r,mut best_rate)=(0usize,0.0f64);
        for r in 2..=n+1 {
            let Some(step_ms)=prices.step_at(r) else {continue};
            let r_rate=Self::expected_from(&cal[..n],r)/(step_ms+push+overhead).max(1.0);
            if r_rate>best_rate {best_r=r;best_rate=r_rate;}
        }
        let probe=self.dir_rounds<2 || self.dir_skipped>=32;
        if best_r>=2 && (best_rate>=rate*0.95 || probe) {
            self.dir_skipped=0;
            if self.dir_rounds<2 && best_rate<rate*1.15 {best_r.min(8)} else {best_r}
        } else {
            self.dir_skipped=self.dir_skipped.saturating_add(1);
            0
        }
    }
    /// Observe the outcome of a direct pre-draft suffix lookup round.
    fn observe_direct_lookup(&mut self,rows:usize,keep:usize) {
        self.dir_rounds=self.dir_rounds.saturating_add(1);
        if rows<=1 {return;}
        let accepted=keep.saturating_sub(1);
        self.dir_first_t=self.dir_first_t*0.92+1.0;
        self.dir_first_h=self.dir_first_h*0.92+f64::from(u8::from(accepted>0));
        for off in 1..(rows-1).min(MAX_POS) {
            if off>accepted {break;}
            self.dir_cont_t=self.dir_cont_t*0.96+1.0;
            self.dir_cont_h=self.dir_cont_h*0.96+f64::from(u8::from(off<accepted));
        }
    }
    /// Number of neural tokens to draft for the next round (`0` = plain step).
    fn propose_depth(&mut self,prices:&Prices,most:usize)->usize {
        self.probing=false;
        if most==0 || !prices.plain_known() {return 0;}
        if self.paused {
            self.plain_run+=1;
            if self.plain_run<self.gap {return 0;}
            self.plain_run=0;self.probing=true;
            return most.min(MIN_PROPOSE);
        }
        self.propose_d.max(MIN_PROPOSE).min(most)
    }
    /// Optimal verify rows `1..=probs.len()+ext_len+1` and Dinkelbach objective value via renewal maximization
    /// over calibrated per-position probabilities (neural + optional zero-cost suffix extension).
    fn verify_rows_val(&self,prices:&Prices,probs:&[f32],ext:Option<(usize,usize)>)->(usize,f64) {
        if probs.is_empty() {return (1,0.0);}
        let ext_len=ext.map_or(0,|(_,len)|len).min(MAX_POS.saturating_sub(probs.len()));
        let total=(probs.len()+ext_len).min(MAX_POS);
        let mut cal=[0.0;MAX_POS];
        for (i,&p) in probs.iter().take(total).enumerate() {cal[i]=self.cal_pos(i,p);}
        let tail_surv=cal[..probs.len().min(total)].iter().product::<f64>();
        let explore_ext=tail_surv>=0.35;
        if let Some((b,_))=ext {
            for off in 0..ext_len {cal[probs.len()+off]=self.ext_cal(b,off,explore_ext);}
        }
        let rate=self.target_rate(prices);
        let push=prices.push.unwrap_or(0.35);
        let base_cost=prices.step_at(1).unwrap_or(28.0);
        let (mut best_r,mut best_val)=(1usize,1.0-rate*base_cost);
        for r in 2..=total+1 {
            let step_ms=prices.step_at(r).unwrap_or(base_cost*(1.10+0.10*(r-1) as f64));
            let val=Self::expected_from(&cal[..total],r)-rate*(step_ms+push);
            if val>best_val {best_r=r;best_val=val;}
        }
        // Prevent calibration freeze if beta[0] starts pessimistic and drops consecutive rounds.
        if best_r==1 && (self.consec_drops>=16 || self.neural_rounds<4) {return (2,best_val);}
        (best_r,best_val)
    }
    #[cfg(test)]
    fn verify_rows(&self,prices:&Prices,probs:&[f32],ext:Option<(usize,usize)>)->usize {
        self.verify_rows_val(prices,probs,ext).0
    }
    /// Survival probability at the end of a neural block (`prod_{i} cal_pos(i, probs[i])`).
    fn tail_survival(&self,probs:&[f32])->f64 {
        probs.iter().enumerate().fold(1.0,|s,(i,&p)|s*self.cal_pos(i,p))
    }
    /// Update logistic estimator, extension stats, depth reach, and pause/probe state after a neural round.
    #[allow(clippy::too_many_arguments)]
    fn observe_round(&mut self,prices:&Prices,probs:&[f32],neural_len:usize,ext_bucket:Option<usize>,rows:usize,keep:usize,round_ms:f64) {
        if rows<=1 {
            self.consec_drops=self.consec_drops.saturating_add(1);
        } else {
            self.consec_drops=0;
            let accepted=keep.saturating_sub(1);
            if accepted>0 {
                self.cond_chain=self.cond_chain*0.92+0.08*(accepted as f64);
            }
            let checked_neural=(rows-1).min(neural_len).min(probs.len()).min(MAX_POS);
            for (s,&p) in probs.iter().enumerate().take(checked_neural) {
                if s>accepted {break;}
                let x=logit(p);
                let p_hat=self.cal_pos(s,p);
                let y=f64::from(u8::from(s<accepted));
                let err=y-p_hat;
                let w=(p_hat*(1.0-p_hat)).max(1.0e-6);
                self.g0+=err*x;self.g[s]+=err;
                self.h00+=w*x*x;self.h0[s]+=w*x;self.h[s]+=w;
                self.n0+=1.0;self.n[s]+=1.0;
            }
            self.fit_rounds=self.fit_rounds.saturating_add(1);
            let target_fit=if self.total_fits<3 {IRLS_EARLY_FIT} else {IRLS_FIT_EVERY};
            if self.fit_rounds>=target_fit {self.refit_irls();}
            if let Some(b)=ext_bucket && rows>neural_len+1 && accepted>=neural_len {
                let checked_ext=(rows-1-neural_len).min(EXT_OFFS);
                let ext_acc=accepted-neural_len;
                for off in 0..checked_ext {
                    if off>ext_acc {break;}
                    self.ext_trials[b][off]=self.ext_trials[b][off]*0.97+1.0;
                    self.ext_hits[b][off]=self.ext_hits[b][off]*0.97+f64::from(u8::from(off<ext_acc));
                }
            }
        }
        let fade=if self.neural_rounds<8 {1.0} else {0.95};
        self.neural_tokens=self.neural_tokens*fade+keep as f64;
        self.neural_ms=self.neural_ms*fade+round_ms.max(1.0);
        self.neural_rounds=self.neural_rounds.saturating_add(1);
        // Deadlock-free streak-gated adaptive wide propose depth (`7 -> 11 -> 15`):
        // - Isolated `keep == 8` hits (common on Code) stay at `d = 7` so `SuffixLookup` handles verbatim copies at
        //   0 ms extra draft cost; a 2-round ceiling streak (`wide_streak >= WIDE_OPEN_STREAK`) or a deep hit
        //   (`keep >= 10`) opens `d = 11`, and `keep >= 11` opens full `d = 15`.
        // - Decaying `deep_ewma` (share of recent rounds keeping `>= 10` neural tokens) identifies sustained deep
        //   workloads (Structured JSON) so they hold `d = 15` across isolated hard tokens (`wide_miss < 3`) and
        //   re-open directly to `15`, while decaying cleanly below `DEEP_REGIME_EWMA` on Code so wide tax stays zero.
        let hit_ceiling=rows>1 && rows>neural_len && keep==rows;
        let hit_base=keep>BASE_PROPOSE || (hit_ceiling && neural_len>=BASE_PROPOSE);
        let deep_hit=f64::from(u8::from(neural_len>BASE_PROPOSE && keep>=10));
        if neural_len>BASE_PROPOSE || !hit_base {
            self.deep_ewma=self.deep_ewma*0.90+0.10*deep_hit;
        }
        let deep_regime=self.deep_ewma>=DEEP_REGIME_EWMA;
        if hit_base {
            self.wide_streak=self.wide_streak.saturating_add(1);
            if deep_regime || self.propose_d>BASE_PROPOSE || self.wide_streak>=WIDE_OPEN_STREAK || keep>=10 {
                self.propose_d=if deep_regime || (neural_len>=11 && keep>=11) || keep>=12 {MAX_POS} else {11};
            }
            self.short_streak=0;self.wide_miss=0;
        } else if self.propose_d>BASE_PROPOSE {
            self.wide_streak=0;
            if rows>BASE_PROPOSE+1 {
                self.wide_miss=0;
            } else if !deep_regime && rows<=5 {
                self.propose_d=BASE_PROPOSE;self.wide_miss=0;
            } else {
                let limit=if deep_regime {3} else {WIDE_COOLDOWN};
                self.wide_miss=self.wide_miss.saturating_add(1);
                if self.wide_miss>=limit {self.propose_d=BASE_PROPOSE;self.wide_miss=0;}
            }
            self.short_streak=0;
        } else if hit_ceiling || rows>=6 {
            self.wide_streak=0;self.propose_d=BASE_PROPOSE;self.short_streak=0;
        } else if rows<=4 {
            self.wide_streak=0;
            self.short_streak=self.short_streak.saturating_add(1);
            if self.short_streak>=16 {self.propose_d=MIN_PROPOSE;}
            else if self.short_streak>=8 {self.propose_d=self.propose_d.min(4);}
        } else {
            self.wide_streak=0;
            self.short_streak=self.short_streak.saturating_sub(1);
        }
        let plain_rate=prices.step_at(1).map_or(0.0,|ms|1.0/ms.max(1.0));
        if std::mem::take(&mut self.probing) {
            let paid=keep as f64/round_ms.max(1.0)>=plain_rate*0.98;
            if paid {
                self.paused=false;self.gap=PROBE_FIRST;
                self.neural_tokens=keep as f64*4.0;self.neural_ms=round_ms.max(1.0)*4.0;self.neural_rounds=4;
            } else {
                self.gap=(self.gap*2).min(PROBE_MOST);
            }
        } else if self.neural_rounds>=FALLBACK_MIN_ROUNDS && self.neural_ms>0.0
            && (self.neural_tokens/self.neural_ms)<plain_rate*0.95 {
            self.paused=true;self.plain_run=0;self.gap=PROBE_FIRST;
        }
    }
}

/// Online exponentially decayed 2-parameter ridge linear fit `y(x) = a + b * x` around a physics prior `(a0, b0)`.
/// Shares statistical strength across all verify widths in a kernel regime (`2..=8` or `9..=16`), eliminating
/// per-width MoE noise cliffs and bandit width starvation while guaranteeing strict monotonicity.
#[derive(Default)]
struct LineFit {sw:f64,sx:f64,sy:f64,sxx:f64,sxy:f64}
impl LineFit {
    fn update(&mut self,x:f64,y:f64) {
        const FADE:f64=0.985;
        self.sw=self.sw*FADE+1.0;
        self.sx=self.sx*FADE+x;
        self.sy=self.sy*FADE+y;
        self.sxx=self.sxx*FADE+x*x;
        self.sxy=self.sxy*FADE+x*y;
    }
    fn solve(&self,a0:f64,b0:f64,wa:f64,wb:f64,a_min:f64,b_min:f64,b_max:f64)->(f64,f64) {
        let (m00,m01,m11)=(self.sw+wa,self.sx,self.sxx+wb);
        let r0=self.sy-a0*self.sw-b0*self.sx;
        let r1=self.sxy-a0*self.sx-b0*self.sxx;
        let det=m00*m11-m01*m01;
        let (da,db)=if det>1.0e-9 {((m11*r0-m01*r1)/det,(m00*r1-m01*r0)/det)} else {(0.0,0.0)};
        ((a0+da).max(a_min),(b0+db).clamp(b_min,b_max))
    }
}

/// Measured prices for one chat: `step1` the plain step, smooth 2-regime affine verify curves for `2..=8` (`lo`,
/// single-block panel entry) and `9..=16` (`hi`, wide few-row panel entry), `draft[1..=16]`, `push`, and `overhead`.
/// First-seen cold graph capture outliers (`step_n[rows] == 1` or samples >30% above smooth prediction) are ignored.
#[derive(Default)]
pub(super) struct Prices {
    step1:Option<f64>,step_n:[u32;17],lo:LineFit,hi:LineFit,
    draft:[Option<f64>;17],pub(super) push:Option<f64>,pub(super) overhead:Option<f64>,
}
impl Prices {
    fn note(slot:&mut Option<f64>,ms:f64) {*slot=Some(slot.map_or(ms,|old|if ms<old {0.55*old+0.45*ms} else {0.82*old+0.18*ms}));}
    /// Concave MoE coupon-collector basis `r^0.70 - 2^0.70` for `rows in 2..=8`: prevents steep `r=2..4`
    /// marginal expert expansion from over-predicting `r=5..8` on Prose and Structured workloads.
    fn lo_x(rows:usize)->f64 {(rows as f64).powf(0.70)-2.0_f64.powf(0.70)}
    pub(super) fn step(&mut self,rows:usize,ms:f64) {
        let Some(n)=self.step_n.get_mut(rows) else {return};
        *n=n.saturating_add(1);
        let seen=*n;
        if rows==1 {
            self.step1=Some(match (self.step1,seen) {
                (None,_)=>ms,
                (Some(old),2) if ms<old=>ms,
                (Some(old),_)=>if ms<old {0.35*old+0.65*ms} else {0.82*old+0.18*ms},
            });
            return;
        }
        // Never fold the first-ever launch of a row width (cold CUDA graph capture) or extreme jitter into the fit.
        if seen<2 {return;}
        let spike_cap=if rows>8 {1.18} else {1.30};
        if let Some(pred)=self.step_at(rows) && ms>pred*spike_cap {return;}
        if rows<=8 {self.lo.update(Self::lo_x(rows),ms);}
        else {self.hi.update((rows-9) as f64,ms);}
    }
    pub(super) fn draft(&mut self,tokens:usize,ms:f64) {if let Some(slot)=self.draft.get_mut(tokens) {Self::note(slot,ms);}}
    pub(super) fn push(&mut self,ms:f64) {Self::note(&mut self.push,ms);}
    pub(super) fn overhead(&mut self,ms:f64) {Self::note(&mut self.overhead,ms.max(0.0));}
    /// True once a plain step (one row) was measured.
    pub(super) fn plain_known(&self)->bool {self.step1.is_some()}
    /// The nearest measured entries below and above `at`, from index `min_i`.
    fn around(table:&[Option<f64>;17],min_i:usize,at:usize)->(Option<(usize,f64)>,Option<(usize,f64)>) {
        let below=(min_i..at).rev().find_map(|i|table[i].map(|ms|(i,ms)));
        let above=(at+1..table.len()).find_map(|i|table[i].map(|ms|(i,ms)));
        (below,above)
    }
    fn lo_params(&self,plain:f64)->(f64,f64) {
        // At r=8, lo_x(8) = 8^0.70 - 2^0.70 ≈ 2.663; prior b0 = plain * 0.24 gives +0.64 * plain from M=2 to M=8.
        self.lo.solve(plain*1.22,plain*0.24,4.0,12.0,plain*1.08,plain*0.08,plain*0.55)
    }
    fn step_at(&self,rows:usize)->Option<f64> {
        let plain=self.step1?.max(1.0e-3);
        if rows<=1 {return Some(plain);}
        let (a_lo,b_lo)=self.lo_params(plain);
        if rows<=8 {return Some(a_lo+b_lo*Self::lo_x(rows));}
        let at8=a_lo+b_lo*Self::lo_x(8);
        let (a_hi,b_hi)=self.hi.solve(at8+plain*0.13,plain*0.068,3.0,10.0,at8+plain*0.04,plain*0.03,plain*0.12);
        Some(a_hi+b_hi*(rows-9) as f64)
    }
    fn draft_at(&self,tokens:usize)->Option<f64> {
        if let Some(ms)=*self.draft.get(tokens)? {return Some(ms);}
        let plain=self.step1?;
        match Self::around(&self.draft,1,tokens) {
            (Some((lo,a)),Some((hi,b)))=>Some(a+(b-a)*(tokens-lo) as f64/(hi-lo) as f64),
            (Some((lo,ms)),None)=>Some(ms+0.25*(tokens-lo) as f64),
            (None,Some((hi,ms)))=>Some((ms-0.25*(hi-tokens) as f64).max(plain*0.06)),
            (None,None)=>Some(plain*0.08+0.25*tokens as f64),
        }
    }
    /// The price of a whole round of `rows` rows, once a plain step was measured.
    pub(super) fn round(&self,rows:usize)->Option<f64> {
        let step=self.step_at(rows)?+self.overhead.unwrap_or(0.0);
        if rows<=1 {return Some(step);}
        Some(step+self.draft_at(rows-1)?+self.push.unwrap_or(0.35))
    }
}

impl<C:Compute,W:Weights<C>> Chat<'_,'_,C,W> {
    pub(super) fn tapped(&mut self,tokens:&[u32],wanted:Wanted,logits:bool,phase:super::super::experts::Phase)->Result<(Vec<u32>,Vec<f32>)> {
        let s=self.service;let c=s.compute;let model=s.model.ok_or("draft model missing")?;
        let active=self.active_slot()?;
        let ctx=self.draft.as_mut().ok_or("draft context missing")?;
        let mut slots=s.slots.borrow_mut();let slot=slots[active].as_mut().ok_or("chat closed")?;
        let cfg=model.config();let ids=cfg.target_layer_ids.iter().map(|&id|id as u64).collect::<Vec<_>>();
        need(ctx.pending.is_empty(),"a draft window is still open")?;
        let decode=phase==super::super::experts::Phase::Decode;
        let verify=decode && tokens.len()>1 && wanted==Wanted::All && !logits;
        // One new token of a chat whose rings are its own: the token pass (`Driver::step_taps`), as a plain chat's.
        let single=decode && !verify && tokens.len()==1 && wanted!=Wanted::None && !slot.shared;
        let started=Instant::now();
        // A plain step that pushes the waiting rows (`LAG_ROWS` of them) is not the price of a plain step.
        let mut flushed=false;
        let result=c.work(|| {
            s.flush_accept(slot)?;
            let (tops,values,taps)=if verify {
                ctx.catch_up(c,model)?;
                let (tops,taps)=s.driver.verify(&mut slot.state,tokens,&ids)?;(tops,Vec::new(),taps)
            } else if single {
                let (step,taps)=s.driver.step_taps(&mut slot.state,tokens[0],&ids)?;
                (vec![step.top],if logits {c.read(&step.logits)?} else {Vec::new()},taps)
            } else {
                let (step,taps)=s.driver.rows_taps_phase(&mut [ChatStep {state:&mut slot.state,tokens,wanted,logits}],phase,&ids,|_,_|Ok(()))?;
                (step.tops,step.logits.as_ref().map(|x|c.read(x)).transpose()?.unwrap_or_default(),taps)
            };
            if verify {ctx.pending=taps;ctx.rows=tokens.len();}
            else if tokens.len()==1 {flushed=ctx.lagged+1==LAG_ROWS;ctx.wait(c,model,&taps)?;}
            else {ctx.catch_up(c,model)?;model.push(c,&mut ctx.chat,&taps,tokens.len())?;}
            Ok((tops,values))
        })?;
        if !verify {slot.shared=false;}
        let ms=started.elapsed().as_secs_f64()*1000.0;
        self.steps+=tokens.len() as u64;self.stepped+=started.elapsed();
        ctx.spent+=ms;ctx.reads+=u32::from(logits);
        if decode {
            if !flushed {ctx.prices.step(tokens.len(),ms);}
            // A drafted round whose draft was dropped (`verify_rows` gave 1 row) ran this plain step and has no check
            // to report it (`truncate`): its outcome is one row kept.
            if !verify && ctx.last_neural {
                let round_ms=ctx.last_draft_ms+ms+ctx.prices.overhead.unwrap_or(0.0);
                ctx.choose.observe_round(&ctx.prices,&ctx.last_probs,ctx.last_neural_len,ctx.last_ext_bucket,1,1,round_ms);
                ctx.last_neural=false;
            }
        }
        Ok(result)
    }
}
impl<C:Compute,W:Weights<C>> Target for Chat<'_,'_,C,W> {
    fn reject_unmapped_greedy(&self)->bool {true}
    fn checkpoint(&self)->usize {self.draft.as_ref().map_or(0,|d|d.read_to()+d.rows)}
    fn truncate(&mut self,pos:usize)->Result<()> {
        let active=self.active_slot()?;
        let s=self.service;let d=self.draft.as_mut().ok_or("draft context missing")?;
        let keep=pos.checked_sub(d.read_to()).ok_or("draft rollback before the window")?;
        need(keep<=d.rows,"draft rollback beyond the window")?;
        if d.pending.is_empty() {return Ok(());}
        let batch_round=d.batch_round.take();
        let mut slots=s.slots.borrow_mut();let slot=slots[active].as_mut().ok_or("chat closed")?;
        if keep==0 {
            slot.pending_accept=None;
            super::super::state::discard(&mut [&mut slot.state]);
        } else if batch_round.is_some() {
            // Defer both target KV/ring accept and draft KV push into the next batched `compute.work` scope,
            // eliminating per-chat `cudaStreamSynchronize` stalls across the cohort.
            need(slot.pending_accept.is_none(),"previous batch accept still pending")?;
            slot.pending_accept=Some((d.rows,keep));
            let taps=std::mem::take(&mut d.pending);
            d.defer_push(taps,keep);
        } else {
            let started=Instant::now();
            s.compute.work(||{s.flush_accept(slot)?;s.driver.accept(&mut slot.state,d.rows,keep)?;
                let m=s.model.ok_or("draft model missing")?;
                let taps=d.pending.iter().map(|v|s.compute.range(v,0,keep*m.config().hidden_size)).collect::<Result<Vec<_>>>()?;
                m.push(s.compute,&mut d.chat,&taps,keep)})?;
            let ms=started.elapsed().as_secs_f64()*1000.0;
            d.prices.push(ms);d.spent+=ms;
        }
        if let Some(t)=d.tally.get_mut(d.rows) {t[1]+=keep as u32;}
        if let Some(round_id)=batch_round {
            if d.last_neural {
                d.choose.observe_multi_chat(&d.last_probs,d.rows,keep);
                d.last_neural=false;
            }
            s.cohort.borrow_mut().finish_chat(round_id,keep);
        } else if d.last_direct_lookup {
            d.choose.observe_direct_lookup(d.rows,keep);
            d.last_direct_lookup=false;
        } else if d.last_neural {
            if d.last_ext_bucket.is_some() && d.rows>d.last_neural_len+1 {
                d.ext_stats[2]+=1;
                d.ext_stats[3]+=keep.saturating_sub(d.last_neural_len+1) as u32;
            }
            let round_ms=d.spent+d.prices.overhead.unwrap_or(0.0);
            d.choose.observe_round(&d.prices,&d.last_probs,d.last_neural_len,d.last_ext_bucket,d.rows,keep,round_ms);
            d.last_neural=false;
        }
        d.last_ext_bucket=None;
        d.pending.clear();d.rows=0;Ok(())
    }
    fn step(&mut self,token:u32)->Result<Vec<f32>> {spark_serve::Session::step(self,token)}
    fn native_draft_policy(&self)->bool {true}
    fn max_verify_rows(&self)->usize {
        if self.draft.is_some() {16.min(self.service.driver.limits().step_rows)} else {1}
    }
    fn model_draft_width(&self)->usize {self.draft.as_ref().map_or(0,|d|d.depth)}
    fn defer_model_draft(&self,limit:usize)->Option<usize> {
        self.draft.as_ref().filter(|d|d.depth>0).map(|_|limit.min(MAX_POS))
    }
    // The request chooses its own depth when it constructs the proposal.
    fn propose_model(&mut self,history:&[u32],anchor:u32,limit:usize)->Result<Option<ModelProposal>> {
        let s=self.service;let m=s.model.ok_or("draft model missing")?;let d=self.draft.as_mut().ok_or("draft context missing")?;
        let cfg=m.config();let started=Instant::now();
        // The round before this one ends here: what its parts did not measure is the overhead of a round.
        if let Some(from)=d.round {d.prices.overhead((started-from).as_secs_f64()*1000.0-d.spent);}
        d.spent=0.0;d.last_neural=false;d.last_direct_lookup=false;d.last_probs.clear();
        d.last_draft_ms=0.0;d.last_neural_len=0;d.last_ext_bucket=None;
        let verify_limit=limit.min(MAX_POS);
        let neural_most=if d.depth>=cfg.gamma() {verify_limit.min(m.wide_depth())} else {verify_limit.min(d.depth)};
        // Online-calibrated 8-token exact suffix match up to 15 tokens (skips drafter only when it beats neural rate).
        let (mut copied,_)=if verify_limit>=2 {d.lookup.propose_after(history,&[anchor],verify_limit,8)} else {(Vec::new(),0)};
        let dir_rows=if copied.len()>=2 {d.choose.direct_lookup_rows(&d.prices,copied.len())} else {0};
        let mut caught=0.0;
        let tokens=if dir_rows>=2 {
            copied.truncate(dir_rows-1);
            d.last_direct_lookup=true;d.ext_stats[0]+=1;
            copied
        } else {
            if copied.len()>=2 {d.ext_stats[1]+=1;}
            let depth=d.choose.propose_depth(&d.prices,neural_most);
            let block_rows=1+depth.max(cfg.gamma());
            if depth==0 || d.read_to()+block_rows>cfg.max_position_embeddings {d.drops[0]+=1;Vec::new()} else {
                // Chain-aware early abort on short-chain chats (saves ~1.36 ms on hopeless drafts; 0.0 on wide chats).
                let abort_below=if depth>1 {d.choose.abort_below(&d.prices)} else {0.0};
                let (mut drafted,mut probs)=s.compute.work(||{d.catch_up(s.compute,m)?;caught=started.elapsed().as_secs_f64()*1000.0;
                    m.propose_confident(s.compute,s.driver.run.weights,&mut d.chat,anchor,&s.driver.embedding(anchor)?,depth,abort_below)})?;
                let draft_ms=started.elapsed().as_secs_f64()*1000.0-caught;
                if drafted.len()==depth {d.prices.draft(depth,draft_ms);} else {d.drops[2]+=1;}
                let mut neural_len=drafted.len();
                // Hybrid zero-cost suffix extension (`M=9..16`): extend past the neural block when confident, AND
                // when wide neural drafting (`neural_len > d.depth`) ran, compete its `7..15` tail against an exact
                // verbatim suffix match after `[anchor, drafted[0..7]]` so verbatim copies never lose to wide neural.
                let mut ext_info=None;
                if verify_limit>neural_len && neural_len>=d.depth && d.choose.tail_survival(&probs)>=0.18 {
                    let mut tail=[0u32;8];
                    let tail_len=if neural_len>=8 {
                        tail.copy_from_slice(&drafted[neural_len-8..neural_len]);
                        8
                    } else {
                        tail[0]=anchor;
                        let copy_n=neural_len.min(7);
                        tail[1..=copy_n].copy_from_slice(&drafted[..copy_n]);
                        copy_n+1
                    };
                    let (ext_toks,match_len)=d.lookup.propose_after(history,&tail[..tail_len],verify_limit-neural_len,1);
                    if let Some(bucket)=ext_bucket(match_len) && !ext_toks.is_empty() {
                        ext_info=Some((bucket,ext_toks.len()));
                        drafted.extend_from_slice(&ext_toks);
                    }
                }
                let (mut pick_rows,best_val)=d.choose.verify_rows_val(&d.prices,&probs,ext_info);
                if neural_len>d.depth && verify_limit>d.depth && d.choose.tail_survival(&probs[..d.depth])>=0.18 {
                    let mut tail=[0u32;8];
                    tail[0]=anchor;
                    let copy_n=d.depth.min(7);
                    tail[1..=copy_n].copy_from_slice(&drafted[..copy_n]);
                    let (ext7_toks,match_len)=d.lookup.propose_after(history,&tail[..=copy_n],verify_limit-d.depth,4);
                    if let Some(bucket)=ext_bucket(match_len) && !ext7_toks.is_empty() {
                        let cand_info=Some((bucket,ext7_toks.len()));
                        let (cand_rows,cand_val)=d.choose.verify_rows_val(&d.prices,&probs[..d.depth],cand_info);
                        if cand_rows>d.depth+1 && cand_val>best_val {
                            drafted.truncate(d.depth);
                            drafted.extend_from_slice(&ext7_toks);
                            probs.truncate(d.depth);
                            neural_len=d.depth;
                            ext_info=cand_info;
                            pick_rows=cand_rows;
                        }
                    }
                }
                drafted.truncate(pick_rows.saturating_sub(1));
                if drafted.is_empty() {d.drops[1]+=1;d.dropped_ms+=draft_ms;}
                d.last_probs=probs;d.last_draft_ms=draft_ms;d.last_neural=true;
                d.last_neural_len=neural_len;
                d.last_ext_bucket=if pick_rows>neural_len+1 {ext_info.map(|(b,_)|b)} else {None};
                drafted
            }
        };
        let ms=started.elapsed().as_secs_f64()*1000.0;
        d.spent+=ms;d.round=Some(started);
        let rows=tokens.len()+1;
        if let Some(t)=d.tally.get_mut(rows) {t[0]+=1;}
        if rows==1 {d.tally[1][1]+=1;}
        Ok(Some(ModelProposal {tokens,proposal_ms:ms,catchup_ms:0.0}))
    }
}

#[cfg(test)]
mod tests {
    use super::{Chooser,FALLBACK_MIN_ROUNDS,MIN_PROPOSE,PROBE_FIRST,Prices};
    #[test]
    fn round_prices_stay_smooth_and_strictly_monotonic_across_all_widths() {
        let mut p=Prices::default();
        p.step(4,60.0);p.draft(3,2.7);
        assert_eq!(p.round(4),None,"no round price before a plain step is measured");
        p.step(1,28.0);p.push(0.35);
        assert_eq!(p.round(1),Some(28.0));
        // Feed realistic verify samples across 2..=16 plus a noisy outlier at r=4.
        for _ in 0..8 {
            p.step(2,34.0);p.step(3,37.0);p.step(5,42.5);p.step(6,45.5);p.step(8,51.0);
            p.step(10,58.0);p.step(12,62.0);p.step(16,70.0);
        }
        p.step(4,55.0);
        // Smooth 2-regime affine fit stays strictly increasing across all 1..=16 with no cliff at r=4.
        for r in 1..16 {
            let a=p.step_at(r).unwrap();let b=p.step_at(r+1).unwrap();
            assert!(a<b,"step_at({r})={a} must stay < step_at({})={b}",r+1);
        }
        assert!(p.step_at(4).unwrap()<45.0,"single noisy sample at r=4 cannot create a cliff");
    }
    #[test]
    fn online_irls_adapts_per_position_and_picks_optimal_depths() {
        let mut p=Prices::default();
        let mut c=Chooser::new();
        assert_eq!(c.propose_depth(&p,7),0,"first round measures plain step");
        // Measured GB10 single-GPU verify timings (ms): plain 27.6, verify 2..=16 = 33.9..63.7.
        p.step(1,27.6);
        for _ in 0..4 {
            for (r,ms) in [(2,33.9),(3,37.0),(4,39.5),(5,41.5),(6,44.0),(7,45.5),(8,48.0),
                (9,52.5),(10,54.5),(11,56.5),(12,58.5),(13,60.0),(14,61.5),(15,62.8),(16,63.7)] {
                p.step(r,ms);
            }
        }
        p.push(0.3);
        c.neural_rounds=10;
        // Low-confidence chain drops to plain (1 row).
        assert_eq!(c.verify_rows(&p,&[0.10,0.10,0.10],None),1);
        // Moderate prose checks 2..4 rows, never wastes 8.
        let r_mid=c.verify_rows(&p,&[0.65,0.45,0.18,0.15,0.15,0.15,0.15],None);
        assert!((2..=3).contains(&r_mid),"moderate prose picks 2..3 rows, got {r_mid}");
        // Code chain checks 5..6 rows.
        let r_code=c.verify_rows(&p,&[0.95,0.92,0.88,0.82,0.72,0.12,0.10],None);
        assert!((5..=6).contains(&r_code),"code picks 5..6 rows, got {r_code}");
        // Structured chain verifies all 8 neural rows, and extends to M=9..16 with confident suffix match.
        let high=[0.99,0.99,0.99,0.99,0.99,0.99,0.99];
        assert_eq!(c.verify_rows(&p,&high,None),8);
        let r_deep=c.verify_rows(&p,&high,Some((0,8)));
        assert!(r_deep>=11,"confident neural + 8-token suffix extends past M=8, got {r_deep}");
        // Near max_tokens tail (`most < MIN_PROPOSE`) must cap cleanly without panicking.
        assert_eq!(c.propose_depth(&p,1),1);
        assert_eq!(c.propose_depth(&p,2),2);
        // Online IRLS refit raises beta when observed hits exceed prior.
        let b0_before=c.beta[0];
        for _ in 0..8 {
            c.observe_round(&p,&[0.80,0.80,0.80,0.80],4,None,5,5,44.0);
        }
        assert!(c.beta[0]>b0_before,"IRLS lifts beta[0] on sustained hits: {} -> {}",b0_before,c.beta[0]);
    }
    #[test]
    fn direct_lookup_gates_out_colliding_templates_and_stretches_verbatim_copies() {
        let mut p=Prices::default();
        p.step(1,27.0);
        for _ in 0..4 {
            for (r,ms) in [(2,33.0),(8,48.0),(9,52.5),(16,64.0)] {p.step(r,ms);}
        }
        p.push(0.3);p.overhead(0.05);
        let mut c=Chooser::new();
        // Fast neural baseline (~110 tok/s = 0.11 tok/ms).
        c.neural_tokens=1100.0;c.neural_ms=10000.0;c.neural_rounds=20;
        let r1=c.direct_lookup_rows(&p,15);
        assert!((2..=8).contains(&r1),"initial probe stays bounded, got {r1}");
        c.observe_direct_lookup(r1,1);
        let r2=c.direct_lookup_rows(&p,15);
        assert!(r2>=2);
        c.observe_direct_lookup(r2,1);
        assert_eq!(c.direct_lookup_rows(&p,15),0,"colliding suffix template yields to neural drafter");
        // Verbatim copy streak stretches direct lookup up to M=16.
        let mut v=Chooser::new();
        v.neural_tokens=800.0;v.neural_ms=10000.0;v.neural_rounds=20;
        for _ in 0..3 {
            let r=v.direct_lookup_rows(&p,15);
            assert!(r>=8);
            v.observe_direct_lookup(r,r);
        }
        assert_eq!(v.direct_lookup_rows(&p,15),16,"verbatim copy stretches to M=16");
    }
    #[test]
    fn sustained_negative_return_pauses_and_probes_for_recovery() {
        let mut p=Prices::default();let mut c=Chooser::new();
        p.step(1,35.0);p.step(2,50.0);p.push(0.5);
        for _ in 0..FALLBACK_MIN_ROUNDS {
            let _=c.propose_depth(&p,7);
            c.observe_round(&p,&[0.2,0.2,0.2],3,None,1,1,45.0);
        }
        assert!(c.paused,"pauses after sustained net loss against plain");
        for _ in 1..PROBE_FIRST {assert_eq!(c.propose_depth(&p,7),0);}
        assert_eq!((c.propose_depth(&p,7),c.probing),(MIN_PROPOSE,true));
        c.observe_round(&p,&[0.95,0.95,0.95,0.95],4,None,5,5,56.0);
        assert!(!c.paused,"profitable probe resumes drafting");
    }
    #[test]
    fn cold_capture_spike_is_filtered_from_smooth_price_curve() {
        let mut p=Prices::default();
        p.step(1,28.0);
        for _ in 0..4 {p.step(2,34.0);p.step(4,40.0);}
        let before=p.step_at(3).unwrap();
        // A single first-seen cold CUDA graph capture spike at r=3 is ignored.
        p.step(3,95.0);
        assert!((p.step_at(3).unwrap()-before).abs()<1e-9);
    }
    #[test]
    fn wide_propose_depth_opens_on_ceiling_hit_and_cools_down_cleanly() {
        let mut p=Prices::default();
        p.step(1,27.0);
        for _ in 0..4 {
            for (r,ms) in [(2,34.0),(8,50.0),(12,62.0),(16,70.0)] {p.step(r,ms);}
        }
        let mut c=Chooser::new();
        assert_eq!(c.propose_depth(&p,15),7,"new chat starts at base depth 7 (0 wide tax on prose)");
        // Prose-like short rounds stay at <= 7.
        for _ in 0..4 {c.observe_round(&p,&[0.6,0.4,0.2,0.1,0.1,0.1,0.1],7,None,3,2,40.0);}
        assert!(c.propose_depth(&p,15)<=7,"short rounds never widen past 7");
        // Single isolated 7/7 hit stays at base depth 7 (0 wide tax on Code); 2nd consecutive ceiling hit opens 11.
        c.observe_round(&p,&[0.98;7],7,None,8,8,54.0);
        assert_eq!(c.propose_depth(&p,15),7,"isolated ceiling hit stays at base depth 7");
        c.observe_round(&p,&[0.98;7],7,None,8,8,54.0);
        assert_eq!(c.propose_depth(&p,15),11,"consecutive ceiling hits open wide depth 11");
        // Deep hits at d=11/15 raise deep_ewma into deep_regime, hold 15 across an isolated hard round, and cool down after 3 short rounds.
        c.observe_round(&p,&[0.98;11],11,None,12,11,64.0);
        assert_eq!(c.propose_depth(&p,15),15,"deep hit at d=11 opens full depth 15");
        for _ in 0..3 {c.observe_round(&p,&[0.98;15],15,None,15,14,70.0);}
        c.observe_round(&p,&[0.6;15],15,None,5,3,48.0);
        assert_eq!(c.propose_depth(&p,15),15,"deep regime survives one isolated hard round");
        for _ in 0..3 {c.observe_round(&p,&[0.6;15],15,None,5,3,48.0);}
        assert_eq!(c.propose_depth(&p,15),7,"sustained short rounds restore base depth 7");
        // In deep_regime, the next ceiling hit jumps directly back to 15 without an intermediate d=11 ramp.
        for _ in 0..2 {c.observe_round(&p,&[0.98;15],15,None,15,14,70.0);}
        for _ in 0..3 {c.observe_round(&p,&[0.6;15],15,None,5,3,48.0);}
        c.observe_round(&p,&[0.98;7],7,None,8,8,54.0);
        assert_eq!(c.propose_depth(&p,15),15,"deep regime re-opens directly to depth 15");
    }
}
