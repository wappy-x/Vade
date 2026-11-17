//! DRAFT CONFIDENCE: how many tokens of a proposal its check pass verifies, by the draft head's own probability of
//! each token and this request's timed passes.
//!
//! MEASURED 2026-11-16 (run draft4-m0-t1, two Sparks, main decc380f49): a check pass of 4 rows costs 84 ms against
//! 41.4 ms for a plain step, and each row past the first costs about 8 ms of routed-expert reads alone. A cycle whose
//! first proposed token is refused still pays the whole pass for one token. At strict depth 2, prose kept 2.11 tokens
//! a cycle and 29 percent of its cycles kept only the first row (d2-t15-four-1): those passes cost 75.5 ms for what a
//! plain step makes in 41.4 ms.
//!
//! So the controller checks the first `k` proposed tokens, the `k` (0 up to all) with the most expected kept tokens
//! beyond what plain steps make in the same time: `E(k) - C(k) / step`, where `E(k) = 1 + q1 + q1 q2 + ...` and
//! `C(k)` is this request's own timed pass of `k + 1` rows (a plain step for `k = 0`). `q` is the chance that a token
//! is kept when the tokens before it are, read from its probability through a calibration that every verdict updates.
//! The proposal itself is paid either way; only the pass is chosen.
//!
//! Nothing here can change a kept token: every token a pass keeps is the target's own greedy choice, and a pass of
//! fewer rows keeps the same tokens as plain steps would.

/// Probability buckets of the calibration, equal widths over 0 to 1.
pub(super) const BUCKETS:usize=10;
/// Pseudo-counts of the prior: a bucket's rate starts at its middle probability and moves to its measured rate.
const PRIOR:f64=4.0;
/// How the calibration forgets: each verdict counts this share of the one after it, so it follows about the last
/// thousand checked tokens of this service.
const DECAY:f64=0.999;
/// The measured chance that a proposed token is kept, given that the tokens before it in its proposal are kept, by
/// the bucket of its probability under the draft head's own row. One calibration for the whole service (controller only).
#[derive(Clone,Debug,PartialEq)]
pub(super) struct Calibration {seen:[f64;BUCKETS],kept:[f64;BUCKETS],
    /// Print only: proposals offered and checked, cycles gated to fewer rows and to a plain step.
    pub(super) offered:u64,pub(super) checked:u64,pub(super) shortened:u64,pub(super) plain:u64}
impl Default for Calibration {fn default()->Self {Self {seen:[0.0;BUCKETS],kept:[0.0;BUCKETS],offered:0,checked:0,shortened:0,plain:0}}}
impl Calibration {
    fn bucket(p:f32)->usize {if p.is_nan() {0} else {((p.clamp(0.0,1.0)*BUCKETS as f32) as usize).min(BUCKETS-1)}}
    /// The chance that a token of probability `p` is kept when the tokens before it are.
    pub(super) fn rate(&self,p:f32)->f64 {
        let b=Self::bucket(p);let middle=(b as f64+0.5)/BUCKETS as f64;(self.kept[b]+PRIOR*middle)/(self.seen[b]+PRIOR)
    }
    /// A verdict: the tokens of probabilities `checked` were verified in order and the first `accepted` were kept. The
    /// kept ones and the first refused one are seen; after a refused token nothing is known.
    pub(super) fn verdict(&mut self,checked:&[f32],accepted:usize) {
        for (at,&p) in checked.iter().enumerate().take(accepted.saturating_add(1)) {
            for b in 0..BUCKETS {self.seen[b]*=DECAY;self.kept[b]*=DECAY;}
            let b=Self::bucket(p);self.seen[b]+=1.0;if at<accepted {self.kept[b]+=1.0;}
        }
    }
    /// One line of the calibration: each bucket's seen count and measured kept share.
    pub(super) fn line(&self)->String {
        (0..BUCKETS).map(|b|format!("p{}={:.0}/{:.2}",b,self.seen[b],if self.seen[b]>0.0 {self.kept[b]/self.seen[b]} else {0.0})).collect::<Vec<_>>().join(" ")
    }
}
/// The tokens of a proposal its pass checks: the count with the most expected kept tokens beyond what plain steps
/// make in the same time (see the top of this file). `rates`: the calibrated chance of each token, in order.
/// `cost(k)`: the ms of the pass that checks `k` tokens (`cost(0)` is a plain step of `step_ms`).
pub(super) fn width(rates:&[f64],step_ms:f64,cost:impl Fn(usize)->f64)->usize {
    if step_ms.is_nan() || step_ms<=0.0 {return rates.len();}
    let (mut best,mut value)=(0usize,1.0-cost(0)/step_ms);let (mut chain,mut expected)=(1.0f64,1.0f64);
    for (at,&rate) in rates.iter().enumerate() {
        chain*=rate.clamp(0.0,1.0);expected+=chain;
        let gain=expected-cost(at+1)/step_ms;if gain>value {best=at+1;value=gain;}
    }
    best
}
/// DRAFT CONFIDENCE: in the token word of a proposal command (9 to 11, 21 up), the chain ends early by its
/// probability (`Verify::propose_mass`). The controller sets it; the worker follows it, so both ranks run one chain.
pub(super) const CHAIN_STOP_FLAG:u32=1<<31;
use super::{Session,Phase,Compute,Exchange,Result,propose_command};
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    /// Proposals of the predictor layer for the tokens after `next`. Empty until this request
    /// has a hidden state of its own (the first call after the prompt).
    /// With them the draft head's probability of each token under its own row (None: the backend made none).
    #[allow(clippy::type_complexity)]
    pub(super) fn propose(&mut self,next:u32,limit:usize)->Result<(Vec<u32>,Vec<Option<f32>>,f64)> {let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}
        let depth=limit.min(self.depth).min(factory.predictor);let stop=self.confident();
        let Some(open)=self.open.as_mut().filter(|open|open.can_propose()) else {return Ok((Vec::new(),Vec::new(),0.0));};
        // A proposal names rows after the open ones: it is cut to the rows the context still has. The predictor layer
        // refuses a chain past its capacity, and that refusal would come after the command is on the wire.
        let depth=depth.min((factory.context as u64).saturating_sub(open.position()) as usize);
        if depth==0 || u64::from(next)>=factory.checkpoint.description().vocab {return Ok((Vec::new(),Vec::new(),0.0));}
        // The layer runs now: the pass after it is not a steady step, whatever its row count.
        let (decoder,id)=(&self.decoder,self.id);self.after_step=false;
        #[allow(clippy::type_complexity)]
        factory.finish((|| -> Result<(Vec<u32>,Vec<Option<f32>>,f64)> {factory.failed.set(true);factory.split.command_for(propose_command(depth),next|if stop {CHAIN_STOP_FLAG} else {0},id)?;
            let tokens=factory.split.work(||{if let Some(groups)=factory.tails.borrow_mut().remove(&id) {open.seed_trail(decoder,&groups)?;}open.propose_mass(decoder,next,depth,stop)})?;
            factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);Ok(tokens)})())
    }
    /// The pass checked the first `drafted` tokens of the open cycle and kept `kept - 1` of them (the cycle's own row first).
    pub(super) fn calibrate(&mut self,drafted:u64,kept:u64) {
        let checked=(drafted as usize).min(self.checked.len());
        if checked>0 && self.checked[..checked].iter().all(|p|!p.is_nan()) {self.factory.confidence.borrow_mut().verdict(&self.checked[..checked],kept.saturating_sub(1) as usize);}
    }
    /// Print only, at a session's end: the service's calibration so far.
    pub(super) fn confidence_report(&self) {
        let c=self.factory.confidence.borrow();
        if c.offered>0 {eprintln!("SERVED CONFIDENCE rank={} session={} offered={} checked={} shortened={} plain={} {}",self.factory.split.topology.rank,self.id,c.offered,c.checked,c.shortened,c.plain,c.line());}
    }
    /// DRAFT CONFIDENCE: whether this request's chains may end early and its passes check only the tokens that pay.
    /// A strict request of a lone chat only: beside another chat a pair decides (pair.rs), and DEFAULT DRAFTING
    /// scores its own full cycles to stop (gate.rs), which shortened cycles would never reach.
    pub(super) fn confident(&self)->bool {self.factory.confidence_on.get() && self.factory.active.get()==1 && self.strict && self.phase==Phase::Off}
    /// DRAFT CONFIDENCE (confidence.rs): how many of a proposal's tokens its pass checks, by the draft head's
    /// probabilities `mass`. All of them when `confident` says no, when a probability is missing, or before this
    /// request has timed a steady step.
    pub(super) fn confident_width(&self,mass:&[Option<f32>])->usize {
        let factory=self.factory;let whole=mass.len();
        let Some(step)=self.step_ms.filter(|_|self.confident() && mass.iter().all(Option::is_some)) else {return whole;};
        let mut calibration=factory.confidence.borrow_mut();
        let rates=mass.iter().map(|p|calibration.rate(p.unwrap_or(0.0))).collect::<Vec<_>>();
        let beside=self.gate.beside_ms();
        let width=width(&rates,step,|k|if k==0 {step} else {self.pass_cost(k+1).unwrap_or(f64::INFINITY)+beside});
        calibration.offered+=1;calibration.checked+=width as u64;
        if width==0 {calibration.plain+=1;} else if width<whole {calibration.shortened+=1;}
        width
    }
}
#[cfg(test)] mod tests {
    use super::*;
    // MEASURED 2026-11-16 on two Sparks (draft4-m0-t1, d2-t15-four-1): plain step 41.4 ms, checks of 3 and 4 rows 75.5
    // and 84.2 ms, a 2-row check about 66 ms (between the two by the request's learned slope).
    const STEP:f64=41.4;
    fn cost(k:usize)->f64 {[STEP,66.0,75.5,84.2][k]}
    #[test] fn a_sure_proposal_is_checked_whole_and_a_hopeless_one_becomes_a_plain_step() {
        assert_eq!(width(&[0.97,0.95,0.95],STEP,cost),3);
        assert_eq!(width(&[0.05,0.9,0.9],STEP,cost),0);
        // Prose at its measured rates (first token kept 0.71, second 0.56 when the first is): the whole depth 2 pays,
        // 1 + 0.71 + 0.40 = 2.11 tokens for 75.5 ms against 1.82 tokens of plain steps.
        assert_eq!(width(&[0.71,0.56],STEP,cost),2);
        // A first token that is likely refused does not pay a pass of 3 rows.
        assert_eq!(width(&[0.3,0.56],STEP,cost),0);
    }
    #[test] fn a_late_doubtful_token_is_left_out_of_the_pass() {
        // Three sure tokens and a fourth that is rarely kept: the 4-row pass costs 8.7 ms for 0.05 tokens.
        assert_eq!(width(&[0.95,0.95,0.06],STEP,cost),2);
    }
    #[test] fn no_timed_step_checks_every_token() {
        assert_eq!(width(&[0.1,0.1],0.0,cost),2);assert_eq!(width(&[0.1,0.1],f64::NAN,cost),2);
    }
    #[test] fn the_calibration_starts_at_the_probability_and_learns_the_kept_share() {
        let mut c=Calibration::default();
        assert!((c.rate(0.95)-0.95).abs()<1e-9 && (c.rate(0.05)-0.05).abs()<1e-9);
        // The head says 0.95 but only half of such tokens are kept: the rate follows the verdicts.
        for _ in 0..200 {c.verdict(&[0.95],1);c.verdict(&[0.95],0);}
        assert!((c.rate(0.95)-0.5).abs()<0.02,"{}",c.rate(0.95));
        // A refused token ends what is seen: the token after it is not counted.
        let mut c=Calibration::default();c.verdict(&[0.55,0.25],0);
        assert!(c.seen[5]>0.0 && c.seen[2]==0.0 && c.kept[5]==0.0);
        c.verdict(&[0.55,0.25],2);assert!(c.kept[5]>0.0 && c.kept[2]>0.0);
        // No probability (a backend that makes none) falls in the lowest bucket and is never used for a choice.
        assert_eq!(Calibration::bucket(f32::NAN),0);assert_eq!(Calibration::bucket(1.0),BUCKETS-1);
    }
}
