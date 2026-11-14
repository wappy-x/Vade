//! DEFAULT DRAFTING: the trial that does not verify. A request that named no draft mode first decodes by its own
//! steps. A few times the predictor layer proposes the next tokens. The proposal is NOT verified by a pass of
//! several rows: the request's own next steps say how many of its tokens were right (a "probe"). Real drafting
//! starts only when every probe was right in full; the first probe that was not ends the trial, and the request
//! goes on by plain steps. A request that does not draft pays its probes only, one of them in most cases. The trial
//! holds no time: the same request gives the same verdict every time. Print and decision input only: no kept token
//! depends on it.
//!
//! WHY SO STRICT (two Sparks, 2026-11-11, the epoch 3 answers, pull request 55's load F7): a cycle of 4 rows costs
//! 134.5 ms against a plain step of 43.3 ms, so it must keep 3.11 rows to pay. Structured text keeps 4 of 4 from its
//! twentieth token to its end and runs 1.23 x plain. Code keeps 2.95 and prose fewer: drafting them loses, also for a
//! strict request that sees the whole prompt. So only text the predictor gets right nearly everywhere should start,
//! and text that does not should leave after one probe: a probe costs about 11 ms, 0.1 percent of 256 tokens.
//! WHY SPREAD: text that does not pay is still right in stretches. The code answer keeps 4 of 4 for 27 tokens after
//! its think block (imports) and then loses; the prose answer keeps its header. Probes side by side would start
//! both. Probes apart read more of the text for the same price.
use std::collections::VecDeque;
/// Tokens of one probe: the base depth of real drafting.
pub const TRIAL_DEPTH:usize=3;
/// Decoded steps before the predictor is seeded or a probe is made: an answer up to TRIAL_START tokens finishes its
/// TRIAL_START-1 decode steps with no seed pass and no probe.
pub const TRIAL_START:u32=64;
/// The kept tokens of a proposal of TRIAL_DEPTH at which a real cycle costs about what plain steps cost (gate.rs
/// DEFAULT STOP counts real cycles with it).
pub(super) const EVEN_TOKENS:i32=2;
/// Probes of a trial. Each must be right in full for real drafting to start.
pub(super) const TRIAL_PROBES:u32=3;
/// THE TRIAL READS THE ANSWER'S OWN TEXT. A chat answer opens inside its think block, and the tokens there say nothing
/// about the text after it. (Epoch 3: every answer opens with a think line of 10 to 16 tokens that repeats the
/// request; there the prose request keeps 4, 4, 3, 3 rows of 4 and the code request 1, 1, 4, 4.) So the trial watches
/// for the token that ends the think block (`Factory::set_answer_marker`) through its start bound, skips ANSWER_SKIP
/// steps after it when it comes near the bound, and never seeds or probes before step TRIAL_START.
pub(super) const THINK_WAIT:u32=TRIAL_START;
/// Steps after the think block (or from the start, for a service that names no such token) before the first probe:
/// the opening of an answer is a header, a fence or a list of imports, and says more than the text will keep.
pub(super) const ANSWER_SKIP:u32=16;
/// Steps between a probe that was right and the next probe (at least TRIAL_DEPTH, so consecutive lookback windows do not overlap).
pub(super) const PROBE_GAP:u32=8;
/// Resolved probes the trial keeps for a reader.
pub(super) const KEPT_SHADOWS:usize=16;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) enum Verdict {Go,Start,GiveUp}
#[derive(Default)]
pub(super) struct Trial {
    /// The last resolved probes as (proposed tokens, matched prefix), newest last.
    results:VecDeque<(u8,u8)>,
    /// Probes made, the ms of the fastest one (print only), and how many were right in full.
    proposals:u32,proposal_low:Option<f64>,score:i32,
    /// Decoded steps seen, steps waited for the end of the think block, whether that wait is over, and steps to skip.
    steps:u32,waited:u32,reading:bool,skip:u32,
    /// The ms the first probe spent on its waiting rows: the prompt and decoded rows, in one pass (print only).
    seed_ms:Option<f64>}
impl Trial {
    /// One of the request's steps gave `token`: whether a probe is due at this step. `marker` is the token that ends
    /// the think block; no probe is due before step TRIAL_START, so an answer of TRIAL_START tokens or fewer pays nothing.
    pub(super) fn due(&mut self,token:u32,marker:Option<u32>)->bool {
        self.steps+=1;
        if !self.reading {
            self.waited+=1;
            match marker {
                // No think block is named: the answer's opening starts at the request's first step.
                None=>{self.reading=true;self.skip=ANSWER_SKIP;},
                // The think block ends at this step: the opening starts at the next one.
                Some(marker) if marker==token=>{self.reading=true;self.skip=ANSWER_SKIP;return false;},
                // The think block did not end in time: the probes read it where they are.
                Some(_) if self.waited>=THINK_WAIT=>self.reading=true,
                Some(_)=>return false,
            }
        }
        if self.skip>0 {self.skip-=1;return false;}
        self.steps>=TRIAL_START
    }
    /// Steps the trial waited for the end of the think block (print only).
    pub(super) fn waited(&self)->u32 {self.waited}
    /// Record a resolved lookback probe of `proposed` tokens that matched `matched` tokens in `ms`, after `waiting_ms`
    /// on the rows that waited for the predictor layer, and say where the trial stands: a probe that was not right in
    /// full, or was shorter than TRIAL_DEPTH, ends it; TRIAL_PROBES right ones start real drafting.
    pub(super) fn record(&mut self,proposed:usize,matched:usize,ms:f64,waiting_ms:f64)->Verdict {
        if proposed==0 {return Verdict::GiveUp;}
        self.seed_ms.get_or_insert(waiting_ms);
        self.proposals+=1;self.proposal_low=Some(self.proposal_low.map_or(ms,|low|low.min(ms)));
        if self.results.len()==KEPT_SHADOWS {self.results.pop_front();}
        self.results.push_back((proposed as u8,matched.min(proposed) as u8));
        if matched<proposed || proposed<TRIAL_DEPTH {return Verdict::GiveUp;}
        self.score+=1;
        if self.score>=TRIAL_PROBES as i32 {Verdict::Start} else {self.skip=PROBE_GAP;Verdict::Go}
    }
    /// The ms of the fastest probe without its waiting rows, and the ms of the first probe's waiting rows (print only).
    pub(super) fn proposal_ms(&self)->f64 {self.proposal_low.unwrap_or(0.0)}
    pub(super) fn seed_ms(&self)->f64 {self.seed_ms.unwrap_or(0.0)}
    pub(super) fn proposals(&self)->u32 {self.proposals}
    /// Probes that were right in full.
    pub(super) fn score(&self)->i32 {self.score}
    /// The resolved probes, oldest first, as (proposed tokens, matched prefix): at most KEPT_SHADOWS.
    pub(super) fn shadows(&self)->impl Iterator<Item=(usize,usize)>+'_ {self.results.iter().map(|&(depth,matched)|(depth as usize,matched as usize))}
    /// Print form of the resolved probes: the matched prefix of each, oldest first.
    pub(super) fn matched(&self)->String {self.results.iter().map(|&(_,matched)|char::from(b'0'+matched.min(9))).collect()}
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn resolved_probes_record_their_matched_prefix_and_first_seed_time() {
        let mut trial=Trial::default();
        assert_eq!(trial.record(3,3,11.0,40.0),Verdict::Go);
        assert_eq!(trial.record(3,1,9.0,0.5),Verdict::GiveUp);
        assert_eq!(trial.record(3,0,12.0,0.5),Verdict::GiveUp);
        assert_eq!(trial.record(0,0,1.0,0.5),Verdict::GiveUp);
        assert_eq!(trial.shadows().collect::<Vec<_>>(),[(3,3),(3,1),(3,0)]);assert_eq!(trial.matched(),"310");
        assert_eq!(trial.proposals(),3);assert_eq!(trial.proposal_ms(),9.0);assert_eq!(trial.seed_ms(),40.0,"the first probe's waiting rows");
    }
    /// The verdicts of a trial whose probes match `matched` tokens in turn.
    fn verdicts(matched:&[usize])->Vec<Verdict> {let mut trial=Trial::default();matched.iter().map(|&matched|trial.record(3,matched,11.0,0.5)).collect()}
    #[test] fn every_probe_must_be_right_in_full_and_the_first_that_is_not_ends_the_trial() {
        use Verdict::{Go,Start,GiveUp};
        assert_eq!(TRIAL_PROBES,3);
        // Structured text: every probe right. Real drafting starts at the last probe.
        assert_eq!(verdicts(&[3,3,3]),[Go,Go,Start]);
        // Text that does not pay leaves at its first probe that is not right in full: one wrong token is enough.
        for first in 0..3 {assert_eq!(verdicts(&[first]),[GiveUp]);}
        // The code answer of epoch 3 keeps its imports and loses after them; the prose answer keeps its header.
        assert_eq!(verdicts(&[3,0]),[Go,GiveUp]);assert_eq!(verdicts(&[3,3,2]),[Go,Go,GiveUp]);assert_eq!(verdicts(&[3,3,0]),[Go,Go,GiveUp]);
        // A proposal shorter than the depth cannot start real drafting at that depth.
        let mut short=Trial::default();assert_eq!(short.record(2,2,7.0,0.5),GiveUp);
        // Every trial ends within TRIAL_PROBES probes, and only a trial of full matches starts.
        for pattern in 0..4usize.pow(TRIAL_PROBES) {
            let matched=(0..TRIAL_PROBES).map(|at|pattern/4usize.pow(at)%4).collect::<Vec<_>>();let all=verdicts(&matched);
            let end=all.iter().position(|verdict|*verdict!=Go).expect("a trial ends");
            assert_eq!(all[end]==Start,matched.iter().all(|&m|m==3),"{matched:?}");assert!(matched[..end].iter().all(|&m|m==3));
        }
    }
    /// The steps (from 1) at which a probe is due for a request whose step `n` gives `token(n)`; each due probe matches in full.
    fn probes(steps:u32,marker:Option<u32>,token:impl Fn(u32)->u32)->Vec<u32> {
        let (mut trial,mut made)=(Trial::default(),Vec::new());
        for step in 1..=steps {
            if trial.due(token(step),marker) {made.push(step);if trial.record(3,3,11.0,0.5)!=Verdict::Go {break;}}
        }
        made
    }
    #[test] fn no_probe_runs_before_trial_start_and_late_markers_still_skip_the_opening() {
        let gap=PROBE_GAP+1;
        // Short answers (up to TRIAL_START tokens = TRIAL_START-1 decode steps) make zero probes, with or without a marker.
        assert!(probes(TRIAL_START-1,None,|_|4).is_empty());
        assert!(probes(TRIAL_START-1,Some(9),|step|if step==12 {9} else {4}).is_empty());
        // Past TRIAL_START: when no marker or an early marker already cleared ANSWER_SKIP, the first probe runs at TRIAL_START.
        assert_eq!(probes(200,None,|_|4),[TRIAL_START,TRIAL_START+gap,TRIAL_START+2*gap]);
        assert_eq!(probes(200,Some(9),|step|if step==12 {9} else {4}),[TRIAL_START,TRIAL_START+gap,TRIAL_START+2*gap]);
        // A marker near the bound still skips ANSWER_SKIP steps after it before probing.
        let late=TRIAL_START-6;let first=late+ANSWER_SKIP+1;
        assert_eq!(probes(200,Some(9),|step|if step==late {9} else {4}),[first,first+gap,first+2*gap]);
        // A think block that never ends starts probing at THINK_WAIT (= TRIAL_START); a marker after it changes nothing.
        assert_eq!(probes(200,Some(9),|_|4)[0],THINK_WAIT);
        assert_eq!(probes(200,Some(9),|step|if step==THINK_WAIT+2 {9} else {4}),probes(200,Some(9),|_|4));
        let mut trial=Trial::default();for _ in 1..TRIAL_START {assert!(!trial.due(4,Some(9)));}
        assert_eq!((trial.proposals(),trial.score(),trial.waited()),(0,0,TRIAL_START-1));
    }
    #[test] fn only_the_last_probes_are_kept() {
        let mut trial=Trial::default();
        for at in 0..KEPT_SHADOWS+3 {let _=trial.record(1,(at+1)%2,1.0,0.5);}
        assert_eq!(trial.shadows().count(),KEPT_SHADOWS);assert_eq!(trial.shadows().next(),Some((1,0)));
    }
}
