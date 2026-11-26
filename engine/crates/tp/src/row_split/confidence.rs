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
/// THE NEW RULES ARE A DEFAULT REQUEST'S ALONE. What a lone chat's request that names no draft mode added when it
/// began to draft from its first passes (Phase::Alone): the chooser of its deepest word, the chain stop of one half,
/// and its verdicts in a kept-share table. A request that NAMES strict takes none of it. It reads and feeds the table
/// `named`, which holds what the one table held before (every request that drafts and is not in Phase::Alone); it
/// sends the proposal word it sent before; the chooser is never told of its cycles. So no number that a default
/// request writes is read by a named one. MEASURED 2026-11-17 (train 25, four graded arms, one table for both): the
/// named strict depth 3 code request of the 256-token load ran 30.359 and 30.511 tokens a second where it ran 31.001
/// and 30.613 before, with 2,770 and 2,614 offered proposals in the table where it had 1,451 and 1,380. The cause of
/// that loss is NOT PROVEN (the request takes one of two paths before this change too, 92 to 95 cycles); this takes
/// away the one way a default request reached a named one.
#[derive(Clone,Debug,Default,PartialEq)]
pub(super) struct Tables {named:Calibration,alone:Calibration,four:Option<Box<[Calibration;2]>>}
impl Tables {
    /// The table of a request: `alone`, it took a default request's rules at its setup (`default_rules`).
    pub(super) fn of(&mut self,alone:bool)->&mut Calibration {if alone {&mut self.alone} else {&mut self.named}}
    /// Four-chat checks start from the existing priors and learn separately from smaller cohorts.
    pub(super) fn checked_for(&mut self,alone:bool,four:bool)->&mut Calibration {
        if !four {return self.of(alone);}
        &mut self.four.get_or_insert_with(||Box::new([self.named.clone(),self.alone.clone()]))[usize::from(alone)]
    }
}
/// Whether a request takes the rules of a lone chat's default drafting: one that STARTS in Phase::Alone. Asked once,
/// at the request's setup (`Session::alone`), and kept to the request's end: a default request that ends its drafting
/// beside another chat (Phase::Over) closes its open cycle after that, and the verdict of that cycle is still its
/// own. It goes into the table `alone`, never into the table of the named requests.
pub(super) fn default_rules(phase:Phase)->bool {phase==Phase::Alone}
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
/// THE DEEPEST WORD (a lone chat's chooser of its proposal depth): the first word it may leave out. The first two words
/// are left to the chain stop and the check pass (`width`).
pub(super) const FIRST_CHOSEN:usize=3;
/// The weight of a cycle's value in a word's running value: about the last eight cycles that drafted the word count.
const WEIGHT:f64=0.125;
/// A PROBE cycle proposes the whole depth, so a word that was left out is measured again. PROBE is the longest wait
/// between two probes, in cycles. MEASURED (draft4-d3-r1, r2, prose): with a probe every 8th cycle the cycles that
/// still drafted word 3 cost 318 and 356 ms of drafting against depth 2; every 32nd cycle drafts a quarter of those.
pub(super) const PROBE:u32=32;
/// The first probe after a word is left out comes PROBE_FIRST cycles later; each probe that leaves a word out again
/// doubles the wait, up to PROBE. MEASURED (dd-1-head, 2026-11-17, a fixed probe every 32nd cycle and a value that
/// started at zero): the third word was left out in 30 of 120 structured cycles and in 113 of 629 code cycles although
/// its value ended at +22.3 and +15.5 ms; on prose (514 of 575 cycles, value -3.0 ms) the rare probe was right.
pub(super) const PROBE_FIRST:u32=4;
/// A word's value in ms before its first verdict: two refusals in a row at a request's start leave the word drafted
/// (4.0, 2.0, 0.3 with a 3.2 ms word and an 8.7 ms row), the third one leaves it out.
const START:f64=4.0;
/// THE DEEPEST WORD: a word deep in a proposal costs a drafting step (about 3.1 ms) in every cycle that drafts it and a
/// checked row (about 9 ms) when its pass checks it; it makes a plain step (about 42 ms) only when it is kept with every
/// word before it. MEASURED (draft4-n2, prose): the third word added 0.21 kept tokens a cycle for 9.7 ms, and depth 3
/// ran 23.8 to 24.0 tokens a second against 24.8 at depth 2. So each request keeps a running value in ms of each word from
/// the third on, from its own cycles: the plain step it made, less its drafting step and its checked row. A cycle
/// proposes up to the first word whose value is under zero, and a probe cycle its whole depth (`PROBE_FIRST`). The depth
/// rides the proposal command, so both ranks draft the same chain. No token changes: a shorter proposal is still verified.
/// `wait`: the cycles from one probe to the next while a word is left out; `since`: the cycles since the last one.
pub(super) struct Chooser {value:[f64;spark_engine::draft::PREDICTOR_DEPTH_MAX+1],cycles:u32,offered:usize,word_ms:f64,cut:u32,wait:u32,since:u32}
impl Default for Chooser {fn default()->Self {Self {value:[START;spark_engine::draft::PREDICTOR_DEPTH_MAX+1],cycles:0,offered:0,word_ms:0.0,cut:0,wait:PROBE_FIRST,since:0}}}
impl Chooser {
    /// The depth of the next proposal, up to `limit`.
    pub(super) fn depth(&mut self,limit:usize)->usize {
        if limit<FIRST_CHOSEN {return limit;}
        self.cycles+=1;
        let mut depth=FIRST_CHOSEN-1;while depth<limit && self.value[depth+1]>=0.0 {depth+=1;}
        // Every word pays: the next word that is left out is probed after the short wait again.
        if depth==limit {self.wait=PROBE_FIRST;self.since=0;return limit;}
        self.since+=1;
        // The probe: the whole depth, and a longer wait for the next one.
        if self.since>=self.wait {self.since=0;self.wait=(self.wait*2).min(PROBE);return limit;}
        self.cut+=1;depth
    }
    fn record(&mut self,word:usize,ms:f64) {if let Some(value)=self.value.get_mut(word) {*value+=WEIGHT*(ms-*value);}}
    /// A proposal of `proposed` words took `proposal_ms`; its pass checks `width` of them. A word drafted and not
    /// checked made nothing.
    pub(super) fn offered(&mut self,proposed:usize,width:usize,proposal_ms:f64) {
        self.offered=proposed;self.word_ms=if proposed==0 {0.0} else {proposal_ms/proposed as f64};
        for word in (width+1).max(FIRST_CHOSEN)..=proposed {self.record(word,-self.word_ms);}
    }
    /// The pass checked `checked` words and kept `accepted` of them; `row[w]`: what checking word `w` added to the pass.
    pub(super) fn verdict(&mut self,checked:usize,accepted:usize,step_ms:f64,row:&[Option<f64>]) {
        for word in FIRST_CHOSEN..=checked.min(self.offered) {
            if let Some(Some(row))=row.get(word) {self.record(word,if accepted>=word {step_ms} else {0.0}-self.word_ms-row);}
        }
        self.offered=0;
    }
    pub(super) fn line(&self)->String {format!("cycles={} cut={} value3={:.1}",self.cycles,self.cut,self.value[FIRST_CHOSEN])}
}
/// DRAFT CONFIDENCE: in the token word of a proposal command (9 to 11, 21 up), the chain ends early by its
/// probability (`Verify::propose_mass`). The controller sets it; the worker follows it, so both ranks run one chain.
pub(super) const CHAIN_STOP_FLAG:u32=1<<31;
/// Beside CHAIN_STOP_FLAG: the request names no draft mode (Phase::Alone), and its chain ends under CHAIN_STOP.
/// Without it the chain ends under CHAIN_STOP_NAMED: a request that NAMES strict sends the word it sent before a
/// request that names no draft mode drafted. MEASURED 2026-11-17 (dd-9, not graded): with one stop of 0.5 for both,
/// named strict depth 3 ran 616 cycles on code where it ran 612 before, 570 on prose where it ran 557.
pub(super) const CHAIN_ALONE_FLAG:u32=1<<30;
/// The flags of a proposal's token word: `confident`, the chain may end early; `alone`, the request names no draft mode.
pub(super) fn stop_flags(confident:bool,alone:bool)->u32 {if !confident {0} else if alone {CHAIN_STOP_FLAG|CHAIN_ALONE_FLAG} else {CHAIN_STOP_FLAG}}
/// A proposal's token word as every rank reads it: the token, and the probability under which its chain ends.
pub(super) fn chain_stop(word:u32)->(u32,Option<f32>) {
    (word&!(CHAIN_STOP_FLAG|CHAIN_ALONE_FLAG),if word&CHAIN_STOP_FLAG==0 {None} else if word&CHAIN_ALONE_FLAG!=0 {Some(CHAIN_STOP)} else {Some(CHAIN_STOP_NAMED)})
}
use super::{Factory,Session,Phase,Compute,Exchange,Result,propose_command};
use crate::bounded_decoder::{CHAIN_STOP,CHAIN_STOP_NAMED};
impl<C:Compute,E:Exchange> Factory<'_,C,E> {
    /// Test control: off, a strict lone chat proposes and checks its whole depth, and a lone chat's request that names
    /// no draft mode runs its trial (trial.rs) instead of drafting as a strict one (`Session::confident`).
    pub fn set_confidence(&self,on:bool) {self.confidence_on.set(on);}
}
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    /// Proposals of the predictor layer for the tokens after `next`. Empty until this request
    /// has a hidden state of its own (the first call after the prompt).
    /// With them the draft head's probability of each token under its own row (None: the backend made none).
    #[allow(clippy::type_complexity)]
    pub(super) fn propose(&mut self,next:u32,limit:usize)->Result<(Vec<u32>,Vec<Option<f32>>,f64)> {let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}
        let Some((depth,flags))=self.chain_plan(next,limit) else {return Ok((Vec::new(),Vec::new(),0.0));};
        let open=self.open.as_mut().ok_or("row split proposal has no open rows")?;
        // The layer runs now: the pass after it is not a steady step, whatever its row count.
        let (decoder,id)=(&self.decoder,self.id);self.after_step=false;
        #[allow(clippy::type_complexity)]
        factory.finish((|| -> Result<(Vec<u32>,Vec<Option<f32>>,f64)> {factory.failed.set(true);factory.split.command_for(propose_command(depth),next|flags,id)?;
            let tokens=factory.split.work(||{if let Some(groups)=factory.tails.borrow_mut().remove(&id) {open.seed_trail(decoder,&groups)?;}open.propose_mass(decoder,next,depth,chain_stop(flags).1)})?;
            factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);Ok(tokens)})())
    }
    /// The depth and the stop flags of the chain `propose` sends after `next`, or None: no chain now (no open rows that
    /// can propose, no depth, or a token outside the vocabulary). ONE PASS (one_pass.rs) asks the same of each chat.
    pub(super) fn chain_plan(&self,next:u32,limit:usize)->Option<(usize,u32)> {let factory=self.factory;
        let depth=limit.min(self.depth).min(factory.predictor);let flags=stop_flags(self.confident(),self.alone);
        // GROUP DRAFTING (pair.rs): the chain of a chat of a group ends by its probability too, under the stop its
        // request has alone (named strict or not).
        let flags=if factory.confidence_on.get() && factory.pair.borrow().chooses(self.id,factory.active.get()) {stop_flags(true,self.alone)} else {flags};
        let open=self.open.as_ref().filter(|open|open.can_propose())?;
        // A proposal names rows after the open ones: it is cut to the rows the context still has. The predictor layer
        // refuses a chain past its capacity, and that refusal would come after the command is on the wire.
        let depth=depth.min((factory.context as u64).saturating_sub(open.position()) as usize);
        if depth==0 || u64::from(next)>=factory.checkpoint.description().vocab.min(u64::from(CHAIN_ALONE_FLAG)) {return None;}
        Some((depth,flags))
    }
    /// The pass checked the first `drafted` tokens of the open cycle and kept `kept - 1` of them (the cycle's own row first).
    pub(super) fn calibrate(&mut self,drafted:u64,kept:u64) {
        let (checked,alone)=((drafted as usize).min(self.checked.len()),self.alone);
        if checked>0 && self.checked[..checked].iter().all(|p|!p.is_nan()) {self.factory.confidence.borrow_mut().checked_for(alone,self.checked_in_four).verdict(&self.checked[..checked],kept.saturating_sub(1) as usize);}
        if let Some(step)=self.step_ms.filter(|_|alone) {
            let row:[Option<f64>;spark_engine::draft::PREDICTOR_DEPTH_MAX+1]=std::array::from_fn(|word|Some(self.pass_cost(word+1)?-self.pass_cost(word)?));
            self.chooser.verdict(drafted as usize,kept.saturating_sub(1) as usize,step,&row);
        }
    }
    /// Print only, at a session's end: the calibration of this request's kind so far (`Tables`).
    pub(super) fn confidence_report(&self) {
        let mut tables=self.factory.confidence.borrow_mut();let c=tables.of(self.alone);
        if c.offered>0 {eprintln!("SERVED CONFIDENCE rank={} session={} offered={} checked={} shortened={} plain={} {} {}",self.factory.split.topology.rank,self.id,c.offered,c.checked,c.shortened,c.plain,c.line(),self.chooser.line());}
    }
    /// DRAFT CONFIDENCE: whether this request's chains may end early and its passes check only the tokens that pay.
    /// A strict request of a lone chat only, named strict or drafting by default (Phase::Alone): beside another chat
    /// the group's prices and the chat's record of the group decide (pair.rs `group_width`), and a trial's real
    /// drafting scores its own full cycles to stop (gate.rs), which shortened cycles would never reach.
    pub(super) fn confident(&self)->bool {self.factory.confidence_on.get() && self.factory.active.get()==1 && self.strict && matches!(self.phase,Phase::Off|Phase::Alone)}
    /// THE DEEPEST WORD: the depth of this cycle's proposal (`Chooser::depth`), for a lone chat's request that names no
    /// draft mode (Phase::Alone). A request that NAMES strict proposes its whole depth in every cycle, as it did before
    /// the chooser: its caller asked for that depth. MEASURED 2026-11-17 (train 23, four arms, the chooser that started
    /// at zero and probed every 32nd cycle): with the chooser on a named request, strict depth 3 ran 39.317 tokens a
    /// second against 41.213 on structured text; the chooser left the third word out in 30 of 120 cycles.
    pub(super) fn chosen_depth(&mut self,limit:usize)->usize {if self.confident() && self.alone {self.chooser.depth(limit)} else {limit}}
    /// The width of the pass (`confident_width`); the chooser learns which words it drafted and left out.
    pub(super) fn checked_width(&mut self,mass:&[Option<f32>],proposal_ms:f64)->usize {
        // GROUP DRAFTING (pair.rs): beside another chat the group's prices choose the rows; the lone chooser learns nothing.
        if self.factory.active.get()>1 {if self.alone {self.chooser.offered(0,0,0.0);}return self.group_width(mass,proposal_ms);}
        let width=self.confident_width(mass);if self.confident() && self.alone {self.chooser.offered(mass.len(),width,proposal_ms);}width
    }
    /// DRAFT CONFIDENCE (confidence.rs): how many of a proposal's tokens its pass checks, by the draft head's
    /// probabilities `mass`. All of them when `confident` says no, when a probability is missing, or before this
    /// request has timed a steady step.
    pub(super) fn confident_width(&self,mass:&[Option<f32>])->usize {
        let factory=self.factory;let whole=mass.len();
        let Some(step)=self.step_ms.filter(|_|self.confident() && mass.iter().all(Option::is_some)) else {return whole;};
        let mut tables=factory.confidence.borrow_mut();let calibration=tables.of(self.alone);
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
    #[test] fn four_chat_verdicts_do_not_train_smaller_cohorts() {
        let mut tables=Tables::default();
        for alone in [false,true] {tables.of(alone).verdict(&[0.9],1);}
        for alone in [false,true] {
            let original=tables.of(alone).clone();
            assert_eq!(*tables.checked_for(alone,true),original);
            tables.checked_for(alone,true).verdict(&[0.9],0);
            assert_eq!(*tables.checked_for(alone,false),original);
            let wide=tables.checked_for(alone,true).clone();
            tables.checked_for(alone,false).verdict(&[0.7],1);
            assert_eq!(*tables.checked_for(alone,true),wide);
        }
    }
    // MEASURED 2026-11-16 on two Sparks (draft4-m0-t1, d2-t15-four-1): plain step 41.4 ms, checks of 3 and 4 rows 75.5
    // and 84.2 ms, a 2-row check about 66 ms (between the two by the request's learned slope).
    const STEP:f64=41.4;
    fn cost(k:usize)->f64 {[STEP,66.0,75.5,84.2][k]}
    #[test] fn a_named_request_sends_the_word_it_sent_before_and_every_rank_reads_the_same() {
        // The controller writes the flags beside the token; the worker reads them back: the same token, the same stop.
        assert_eq!(chain_stop(77|stop_flags(false,true)),(77,None));
        assert_eq!(chain_stop(77|stop_flags(false,false)),(77,None));
        assert_eq!(chain_stop(77|stop_flags(true,false)),(77,Some(CHAIN_STOP_NAMED)));
        assert_eq!(chain_stop(77|stop_flags(true,true)),(77,Some(CHAIN_STOP)));
        assert_eq!(chain_stop((CHAIN_ALONE_FLAG-1)|stop_flags(true,true)).0,CHAIN_ALONE_FLAG-1);
        // A request that names strict sets the one bit it set before a default request drafted; only a default request sets the second.
        assert_eq!(stop_flags(true,false),CHAIN_STOP_FLAG);assert_eq!(stop_flags(true,true),CHAIN_STOP_FLAG|CHAIN_ALONE_FLAG);
    }
    #[test] fn only_a_request_that_names_no_draft_mode_takes_the_new_rules() {
        // Phase::Alone is the lone chat's request that names no draft mode; a request that names strict is Phase::Off.
        assert!(default_rules(Phase::Alone));
        for phase in [Phase::Off,Phase::Trial,Phase::Drafting,Phase::Over] {assert!(!default_rules(phase));}
    }
    #[test] fn a_default_request_never_moves_the_table_of_a_named_one() {
        let mut tables=Tables::default();let before=tables.of(false).clone();
        // Five hundred refused tokens of a default request: its own table falls, the named one is as it was.
        for _ in 0..500 {tables.of(true).verdict(&[0.95,0.95,0.95],0);}
        assert!(tables.of(true).rate(0.95)<0.1);assert_eq!(*tables.of(false),before);assert_eq!(tables.of(false).rate(0.95),before.rate(0.95));
        // And a named request never moves the table of a default one.
        let alone=tables.of(true).clone();for _ in 0..500 {tables.of(false).verdict(&[0.55],1);}
        assert_eq!(*tables.of(true),alone);assert!(tables.of(false).rate(0.55)>0.9);
    }
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
    // What checking word w adds to a pass (`Chooser::verdict`): the same request's measured rows.
    const ROWS:[Option<f64>;spark_engine::draft::PREDICTOR_DEPTH_MAX+1]=[None,Some(24.6),Some(9.5),Some(8.7),Some(8.7),Some(8.7),None,None];
    #[test] fn the_chooser_leaves_out_a_third_word_that_does_not_pay_and_tries_it_again() {
        let mut c=Chooser::default();
        // Under the third word nothing is chosen, and no cycle is counted.
        assert_eq!((c.depth(2),c.depth(1),c.cycles),(2,1,0));
        // Nothing measured yet: the whole depth. Three words drafted in 9.6 ms and checked, two kept: the third paid
        // its step and its row for nothing. Two such cycles at the start leave the word drafted; the third leaves it out.
        for refused in 1..=3 {assert_eq!(c.depth(3),3,"refusal {refused}");c.offered(3,3,9.6);c.verdict(3,2,STEP,&ROWS);assert_eq!(c.value[3]<0.0,refused==3,"{}",c.value[3]);}
        // The probes come after 4 and then 8 cycles while the word stays unpaid; a probe cycle drafts the whole depth.
        for wait in [PROBE_FIRST,2*PROBE_FIRST] {
            for cycle in 1..wait {assert_eq!(c.depth(3),2,"wait {wait} cycle {cycle}");}
            assert_eq!(c.depth(3),3,"probe after {wait}");c.offered(3,3,9.6);c.verdict(3,2,STEP,&ROWS);
        }
        // The probe after 16 cycles keeps its third word: the word pays again, and the next wait is the short one.
        for cycle in 1..4*PROBE_FIRST {assert_eq!(c.depth(3),2,"cycle {cycle}");}
        assert_eq!(c.depth(3),3);c.offered(3,3,9.6);c.verdict(3,3,STEP,&ROWS);assert!(c.value[3]>0.0,"{}",c.value[3]);
        assert_eq!(c.depth(3),3);assert_eq!((c.cut,c.wait,c.since),(7*PROBE_FIRST-3,PROBE_FIRST,0));
        // A third word drafted and left out of the pass made nothing: it costs its step alone.
        let before=c.value[3];c.offered(3,2,9.6);c.verdict(2,2,STEP,&ROWS);
        assert!((c.value[3]-(before+WEIGHT*(-3.2-before))).abs()<1e-9,"{}",c.value[3]);
    }
    #[test] fn the_chooser_waits_longer_from_probe_to_probe_while_a_word_stays_unpaid() {
        // A third word that is drafted and never checked (prose): the waits double from PROBE_FIRST up to PROBE and stay there.
        let mut c=Chooser::default();c.value[3]= -50.0;
        for wait in [PROBE_FIRST,8,16,PROBE,PROBE] {
            for cycle in 1..wait {assert_eq!(c.depth(3),2,"wait {wait} cycle {cycle}");}
            assert_eq!(c.depth(3),3,"probe after {wait}");c.offered(3,2,9.6);c.verdict(2,2,STEP,&ROWS);assert!(c.value[3]<0.0);
        }
        assert_eq!((c.wait,c.cycles,c.cut),(PROBE,PROBE_FIRST+8+16+2*PROBE,PROBE_FIRST+8+16+2*PROBE-5));
    }
    #[test] fn the_chooser_proposes_up_to_the_first_word_that_does_not_pay() {
        let mut c=Chooser::default();
        // Five words checked, three kept, three cycles in a row: the third word paid, the fourth and fifth did not.
        for cycle in 1..=3 {assert_eq!(c.depth(5),5,"cycle {cycle}");c.offered(5,5,16.0);c.verdict(5,3,STEP,&ROWS);}
        assert!(c.value[3]>0.0 && c.value[4]<0.0 && c.value[5]<0.0);
        assert_eq!(c.depth(5),3);
        // A chain that the chain stop ended at two words measures no deeper word, and a verdict without an offer counts nothing.
        let value=c.value;c.offered(2,2,6.4);c.verdict(2,2,STEP,&ROWS);c.verdict(5,5,STEP,&ROWS);assert_eq!(c.value,value);
    }
}
