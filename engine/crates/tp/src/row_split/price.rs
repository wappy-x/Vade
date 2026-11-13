//! How deep a drafting request proposes on the two-Spark text service: by its own measured record.
//!
//! The Speculator's chooser compares the expected tokens a ms of a plain step and of a pass, from
//! this request's times and hits. Three times on 2026-11-11 that price model sent requests that
//! keep their proposals back to plain steps:
//! - Early each input is one or two samples and the compared rates sat under 3 percent apart, so a
//!   plain step 1 ms faster tipped requests to plain steps (structured at strict depth 3 fell from
//!   27.2 to 24.8 tok/s). Rule one below answered that: the first cycles propose the whole depth.
//! - On the epoch 3 answers the first cycles sit in free text that opens each answer and lose
//!   rows there (code kept 1, 1, 4, 4 rows of 4; structured 1, 4, 3, 2). The chooser then left for
//!   plain steps and stayed, while later proposals were kept as before (structured kept 14 of 14
//!   tries): 21.6 tok/s against 22.4 plain and 27.4 before. A request that goes back makes no
//!   proposal that could correct its prices; the chooser's own tries are one-token proposals, a
//!   2-row pass that costs more than two plain steps even when it is kept.
//!
//! So the price model can no longer take a request of depth 2 or more off its whole depth. Such a
//! request proposes its whole depth
//! 1. in its first THIN_CYCLES proposing cycles;
//! 2. after a cycle that kept its whole proposal;
//! 3. while its record pays: its recent proposing cycles, measured, made tokens at PAY_FLOOR of
//!    its plain rate or more;
//! 4. when the caller asks for the whole depth itself (the chooser when its prices say so, and
//!    the tests and the check tool that drive a session by hand).
//!
//! Otherwise it takes plain steps and tries one whole proposal after TRY_WAIT_MIN plain steps. A try
//! starts a new record. When the cycles a try started end, the next try waits TRY_WAIT_MIN plain
//! steps if those cycles as a whole paid, and twice as long as the last one if not, up to
//! TRY_WAIT_MAX. It never makes a proposal shorter than its whole depth. A margin against back-off
//! (the plain step priced 8 percent up) was measured and taken out: it kept prose drafting at a loss.
//! Nothing here can change a kept token: every proposed row is still verified.
/// Proposing cycles a request begins with its whole depth whatever its record says. Four cycles
/// give the record four samples.
pub(super) const THIN_CYCLES:u32=4;
/// Whether a request of `whole` proposals a cycle that has begun `begun` proposing cycles is still
/// in its first ones. A request of depth 1 is left out: its pass of two rows costs more than two
/// plain steps even when the proposal is kept.
pub(super) fn thin(whole:usize,begun:u32)->bool {whole>=2 && begun<THIN_CYCLES}
/// Plain steps before a try: at first, and again after cycles that paid.
pub(super) const TRY_WAIT_MIN:u32=8;
/// The longest wait before a try, in plain steps.
pub(super) const TRY_WAIT_MAX:u32=64;
/// How the record forgets: a proposing cycle counts 31/32 of the one after it, so the record is
/// about the last 32 cycles. A shorter record left text that pays 7 percent (code, 3.3 rows kept of
/// 4) on a bad run of six cycles; this one still follows text that turns to prose in about ten.
const DECAY:f64=31.0/32.0;
/// A request keeps proposing while its record makes tokens at this share of its plain rate or more.
const PAY_FLOOR:f64=0.97;
/// The measured record of one request's proposing cycles and what it decides.
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub(super) struct Pay {
    /// Decayed sums over the proposing cycles since the record began: rows kept (the first row of
    /// each pass included) and ms.
    kept:f64,ms:f64,
    /// The last proposing cycle drafted the request's whole depth and kept every row.
    whole:bool,
    /// Plain steps since the last proposing cycle, and how many a try waits for (0 reads as TRY_WAIT_MIN).
    idle:u32,wait:u32,
    /// The cycles since the last try, while they go on: their rows kept and their ms, not decayed.
    run:Option<(f64,f64)>,
    /// What the last call of `depth` was given: the request's whole depth and its plain step, ms.
    depth:usize,step_ms:f64,
    /// For the print: proposals made after a whole keep, on a record that pays, on the caller's
    /// word, and as a try; tries whose cycles paid as a whole.
    held:u32,paid:u32,asked:u32,tries:u32,tries_paid:u32}
impl Pay {
    /// Whether the record makes tokens at PAY_FLOOR of the plain rate or more.
    fn pays(&self)->bool {self.ms>0.0 && self.kept*self.step_ms>=PAY_FLOOR*self.ms}
    /// The depth of the next proposal of a request whose whole depth is `whole`, that has begun
    /// `begun` proposing cycles, whose plain step takes `step_ms`. `pick` is the caller's depth:
    /// a request of depth 1 or 0 takes it; a deeper one takes it only when it is the whole depth.
    pub(super) fn depth(&mut self,whole:usize,begun:u32,pick:usize,step_ms:f64)->usize {
        self.depth=whole;self.step_ms=step_ms;
        if whole<2 {return pick;}
        if thin(whole,begun) {return whole;}
        if self.whole {self.held+=1;return whole;}
        if self.pays() {self.paid+=1;return whole;}
        if pick>=whole {self.asked+=1;return whole;}
        // The request is on plain steps. The cycles its last try started are over: they set the wait.
        // (A try that made no cycle, because the request had no room for a proposal, sets nothing.)
        if let Some((kept,ms))=self.run.take().filter(|run|run.1>0.0) {
            if kept*step_ms>=ms {self.tries_paid+=1;self.wait=TRY_WAIT_MIN;} else {self.wait=(self.wait.max(TRY_WAIT_MIN)*2).min(TRY_WAIT_MAX);}
        }
        if self.idle<self.wait.max(TRY_WAIT_MIN) {return 0;}
        // A try: one whole proposal on a new record.
        self.tries+=1;self.kept=0.0;self.ms=0.0;self.run=Some((0.0,0.0));whole
    }
    /// A plain step ran.
    pub(super) fn step(&mut self) {self.idle=self.idle.saturating_add(1);}
    /// A proposing cycle ended: its pass verified `drafted` proposed rows and `kept` rows were kept,
    /// the first row included, in `ms`.
    pub(super) fn verdict(&mut self,drafted:u64,kept:u64,ms:f64) {
        self.kept=self.kept*DECAY+kept as f64;self.ms=self.ms*DECAY+ms;self.idle=0;
        self.whole=self.depth>=2 && drafted>=self.depth as u64 && kept>drafted;
        if let Some(run)=self.run.as_mut() {run.0+=kept as f64;run.1+=ms;}
    }
    /// A proposal had every row cut before its pass (a stop token, a token the request may not
    /// make): the step made one token at the cost of the proposal too. It counts as a cycle that
    /// kept one row, so a request whose proposals keep being cut leaves for plain steps.
    pub(super) fn unused(&mut self,ms:f64) {
        let ms=ms+self.step_ms;self.kept=self.kept*DECAY+1.0;self.ms=self.ms*DECAY+ms;self.idle=0;self.whole=false;
        if let Some(run)=self.run.as_mut() {run.0+=1.0;run.1+=ms;}
    }
    /// One line for a request of depth 2 or more: how its proposals were decided.
    pub(super) fn report(&self,rank:u32,session:u64,depth:usize) {
        if depth>=2 {eprintln!("SERVED PAY rank={rank} session={session} depth={depth} after_whole_keep={} on_paying_record={} on_callers_word={} tries={} tries_paid={} wait={} record_rows={:.2} record_ms={:.1} pays={}",
            self.held,self.paid,self.asked,self.tries,self.tries_paid,self.wait.max(TRY_WAIT_MIN),self.kept,self.ms,u8::from(self.pays()));}
    }
}
#[cfg(test)] mod tests {
    use super::*;
    // MEASURED 2026-11-11 on two Sparks, epoch 3 main (load JC): plain step 44.67 ms, 4-row pass 123.17 ms,
    // about 13 ms of proposal, waiting rows and verdict beside it.
    const STEP:f64=44.67;const CYCLE:f64=136.3;
    /// A request of depth 3 that makes `tokens` tokens; `rows(i)` is what its i-th whole cycle keeps
    /// (1 to 4 rows). Returns the plain steps, the whole cycles, and the request's ms.
    fn request(tokens:usize,rows:impl Fn(usize)->u64)->(usize,usize,f64,Pay) {
        let (mut pay,mut begun,mut made,mut plain,mut cycles,mut ms)=(Pay::default(),0u32,2usize,2usize,0usize,2.0*STEP);
        // The first two steps of a request are plain: the second one times the plain step.
        while made<tokens {
            // The chooser backs off in every cycle: it cannot take a request of depth 3 off its whole depth.
            let depth=pay.depth(3,begun,0,STEP);assert!(depth==0 || depth==3,"a request of depth 3 proposed {depth}");
            if depth==0 {pay.step();made+=1;plain+=1;ms+=STEP;continue;}
            let kept=rows(cycles).min((tokens-made) as u64);pay.verdict(3,kept,CYCLE);begun+=1;cycles+=1;made+=kept as usize;ms+=CYCLE;
        }
        (plain,cycles,ms,pay)
    }
    #[test] fn a_request_that_lost_rows_in_its_first_tokens_comes_back_after_eight_plain_steps() {
        // Structured on the epoch 3 answers: the four first cycles kept 1, 4, 3, 2 rows; after them every proposal is kept.
        let start=[1,4,3,2];let (plain,cycles,ms,pay)=request(256,|i|start.get(i).copied().unwrap_or(4));
        // Two plain steps at the start, eight before the one try, none after it.
        assert_eq!((plain,cycles,pay.tries),(10,63,1));
        // On main the same request took 216 plain steps (21.6 tok/s against 22.4 plain).
        assert!(256.0/ms*1000.0>28.0,"{}",256.0/ms*1000.0);
    }
    #[test] fn text_that_keeps_every_proposal_never_leaves_its_whole_depth() {
        let (plain,cycles,_,pay)=request(256,|_|4);assert_eq!((plain,cycles,pay.tries),(2,64,0));
    }
    #[test] fn text_that_loses_by_drafting_gets_a_few_whole_tries_and_no_one_token_proposal() {
        // Prose keeps one proposal of three here: 2 rows a cycle, 136 ms against 89 ms of plain steps.
        let (plain,cycles,ms,pay)=request(256,|_|2);
        // Four first cycles, then tries after 8, 16, 32, 64 and 64 plain steps.
        assert_eq!((cycles,pay.tries,pay.tries_paid),(9,5,0));assert_eq!(plain,2+8+16+32+64+64+52);
        // It stays within 4 percent of its plain rate (main's two-row tries cost the same request about 4 percent).
        assert!(256.0/ms*1000.0>0.96*1000.0/STEP);
    }
    #[test] fn text_that_pays_seven_percent_stays_through_a_bad_run() {
        // Code, MEASURED before epoch 3 (pair one, 75 cycles of one request): rows kept by each 4-row cycle, 3.3 on
        // average, with a run of six cycles that kept 1, 4, 1, 1, 1, 2. The epoch 3 start (1, 1, 4, 4) comes first.
        let code=[1,1,4,4, 4,4,4,4,3,4,4,4,4,4,4,3,2,2,4,4,4,2,4,3,3,4,4,4,2,2,3,4,4,4,4,3,4,4,1,4,1,1,1,2,4,2,4,4,4,2,4,3,2,1,3,4,4,4,2,4,1,4,4,3,4,3,4,4,4,4,4,4,4,4,4,4,4,2,4];
        let (plain,_,ms,pay)=request(256,|i|code[i%code.len()]);
        assert!(plain<=10 && pay.tries<=1,"{plain} plain steps, {} tries",pay.tries);
        assert!(256.0/ms*1000.0>1.06*1000.0/STEP,"{}",256.0/ms*1000.0);
    }
    #[test] fn the_wait_before_a_try_follows_what_the_cycles_of_the_last_try_made() {
        let mut pay=Pay::default();
        // Four first cycles that keep only their first row: the request leaves for plain steps.
        for begun in 0..4 {assert_eq!(pay.depth(3,begun,0,STEP),3);pay.verdict(3,1,CYCLE);}
        // The plain steps the request takes before its next try (the try is begun when this returns).
        let waited=|pay:&mut Pay| {let mut steps=0;while pay.depth(3,9,0,STEP)==0 {pay.step();steps+=1;assert!(steps<=TRY_WAIT_MAX);}steps};
        // One-row cycles until the request leaves: how many it took.
        let lost=|pay:&mut Pay| {let mut cycles=0;loop {pay.verdict(3,1,CYCLE);cycles+=1;if pay.depth(3,9,0,STEP)==0 {return cycles;}}};
        assert_eq!(waited(&mut pay),TRY_WAIT_MIN);
        // The try keeps 2 rows: 89 ms of plain steps for 136 ms. Its cycles lost: the next try waits twice as long.
        pay.verdict(3,2,CYCLE);assert_eq!(waited(&mut pay),16);
        // The try and three more cycles keep their whole proposal, then two keep one row: the record falls under its
        // floor. 18 rows in 6 cycles are 804 ms of plain steps for 818 ms: lost as a whole, by little. 32 plain steps.
        for _ in 0..4 {pay.verdict(3,4,CYCLE);assert_eq!(pay.depth(3,9,0,STEP),3);}
        assert_eq!(lost(&mut pay),2);assert_eq!(waited(&mut pay),32);
        // Thirty whole cycles, then nine that keep one row before the record is under its floor: 129 rows in 39
        // cycles are 5,762 ms of plain steps for 5,316 ms. Those cycles paid: the next try comes after 8 plain steps.
        for _ in 0..30 {pay.verdict(3,4,CYCLE);assert_eq!(pay.depth(3,9,0,STEP),3);}
        assert_eq!(lost(&mut pay),9);assert_eq!(waited(&mut pay),TRY_WAIT_MIN);
        assert_eq!((pay.tries,pay.tries_paid),(4,1));
    }
    #[test] fn a_request_of_depth_one_and_a_request_that_does_not_draft_take_the_choosers_depth() {
        let mut pay=Pay::default();
        for pick in 0..=1 {for begun in 0..9 {assert_eq!(pay.depth(1,begun,pick,STEP),pick);assert_eq!(pay.depth(0,begun,pick,STEP),pick);}}
        // A deeper request never makes the chooser's one-token or two-token proposal; a caller that asks for the
        // whole depth gets it (the check tool and the tests that drive a session by hand).
        for begun in 0..4 {pay.depth(3,begun,1,STEP);pay.verdict(3,1,CYCLE);}
        for pick in 0..=2 {assert_eq!(pay.depth(3,4,pick,STEP),0);}
        for pick in 3..=4 {assert_eq!(pay.depth(3,4,pick,STEP),3);}
        assert_eq!(pay.asked,2);
    }
    #[test] fn proposals_that_are_cut_before_their_pass_count_and_end_the_first_cycles() {
        assert!(thin(3,0) && thin(2,3) && !thin(3,4) && !thin(1,0) && !thin(0,0));
        // Every proposal is cut (the request may not make the predictor's first token): four begun cycles end the
        // first cycles, and each cut proposal counts as a cycle that kept one row at the cost of a step and a proposal.
        let mut pay=Pay::default();
        for begun in 0..4 {assert_eq!(pay.depth(3,begun,0,STEP),3);pay.unused(11.0);}
        assert_eq!(pay.depth(3,4,0,STEP),0,"one row in 55.7 ms does not pay against a step of 44.67 ms");
        for _ in 0..7 {pay.step();assert_eq!(pay.depth(3,4,0,STEP),0);}
        pay.step();assert_eq!(pay.depth(3,4,0,STEP),3);pay.unused(11.0);
        // The try was cut too: the next one waits 16 plain steps.
        for _ in 0..16 {assert_eq!(pay.depth(3,5,0,STEP),0);pay.step();}
        assert_eq!(pay.depth(3,5,0,STEP),3);
    }
    #[test] fn the_record_pays_on_code_and_not_on_prose_at_the_days_times() {
        // Code keeps 3.27 rows a 4-row cycle on the epoch 3 answers (26 cycles of one request), prose about 2.4.
        let record=|rows:f64|Pay {kept:rows,ms:CYCLE,step_ms:STEP,..Pay::default()};
        assert!(record(3.27).pays() && record(4.0).pays() && !record(2.4).pays() && !record(2.9).pays());
        // Three rows a cycle is the edge: 134.0 ms of plain steps against 136.3 ms.
        assert!(record(3.0).pays() && 3.0*STEP<CYCLE);
    }
}
