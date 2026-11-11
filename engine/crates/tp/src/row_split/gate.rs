//! The proposer's account of one request: what its proposing cycles cost and kept. Print and
//! chooser input only: no kept token depends on it.
use std::time::Instant;
/// The cycle that is open: a proposal was made and its verdict has not come yet.
pub(super) struct Cycle {
    /// Its four parts so far, ms: proposal, waiting rows, pass, verdict.
    parts:[f64;4],
    /// The position at its entry, and its start.
    entry:u64,started:Instant,
    /// The proposal's length: the tokens the predictor layer made. Its cost is `parts[0]`.
    proposed:u64,
    /// The draft's length: the proposed rows its pass verified. The scheduler may cut a
    /// proposal (a stop token, a token the tokenizer does not hold), so this can be shorter.
    drafted:u64,
    /// Kept rows that waited for the layer before this proposal, but the last.
    waited:u64}
/// What the account has counted, for the check tools and the tests. Sums cover the same cycles
/// as their counts: `proposed_tokens`, `drafted_rows`, `kept_rows` and the ms sums are over
/// `cycles`; a proposal whose cycle had no verdict is in the three `unused` fields and nowhere else.
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct Account {
    /// Proposing cycles that reached a verdict.
    pub cycles:u32,
    /// Tokens the predictor layer made in those cycles (the proposal length).
    pub proposed_tokens:u64,
    /// Proposed rows the passes of those cycles verified (the draft length).
    pub drafted_rows:u64,
    /// Rows those cycles kept, the first row of each pass included.
    pub kept_rows:u64,
    /// Sum of the proposal part of those cycles, ms.
    pub proposal_ms:f64,
    /// Proposals whose rows never reached a verdict (every row cut, or the pass failed), their tokens, their ms.
    pub unused_proposals:u32,pub unused_tokens:u64,pub unused_ms:f64,
    /// This request's passes of one row, and those the head answered with the token alone.
    pub one_row_passes:u64,pub picked_on_device:u64}
/// A proposing cycle is a proposal (with the rows that waited for the layer), its verify pass
/// and its verdict; the account keeps what such cycles cost and kept. The depth of each proposal
/// is chosen by the Speculator's native chooser (cumulative hits for each depth, this account's
/// measured costs). The very first call makes no proposal, so that a plain step of this request
/// is timed. Nothing here can change a kept token.
#[derive(Default)]
pub(super) struct Gate {pub(super) tried:u32,pub(super) kept:f64,pub(super) cycle_ms:f64,pub(super) declined:u32,
    open:Option<Cycle>,
    /// Sums over the proposing cycles, ms: proposal, waiting rows, pass, verdict.
    pub(super) parts:[f64;4],
    /// The first proposal's waiting rows are the prompt rows: paid once, kept out of the means.
    pub(super) seed_ms:f64,
    /// Sum over the proposing cycles, ms: the wall time from the proposal's start to the verdict's
    /// end that is in none of the four parts (the speculator, the scheduler, the stream).
    edge_ms:f64,
    /// Verdicts that kept only a part of their pass, and their ms: these rebuild the recurrent state.
    partial:u32,partial_ms:f64,
    /// Over the proposing cycles: proposed tokens, proposed rows their passes verified, and kept
    /// rows that waited for the layer (the first proposal's are prompt rows).
    proposed:u64,drafted:u64,waited:u64,
    /// Proposals whose cycle had no verdict, their tokens and their ms (proposal and waiting rows).
    unused:u32,unused_tokens:u64,unused_ms:f64,
    /// This request's passes of one row, and those the head answered with the token alone.
    steps:u64,picked:u64}
impl Gate {
    pub(super) fn allows(&mut self,step_ms:Option<f64>)->bool {if step_ms.is_none() {self.declined+=1;}step_ms.is_some()}
    /// A proposal call starts. A cycle that is still open had no verdict: it is counted as unused, not dropped.
    pub(super) fn begin(&mut self) {
        if let Some(cycle)=self.open.take() {self.unused+=1;self.unused_tokens+=cycle.proposed;self.unused_ms+=cycle.parts[0]+cycle.parts[1];}
    }
    /// The proposal made `proposed` tokens in `proposal_ms`, after `waiting_ms` on `waited` rows.
    pub(super) fn proposed(&mut self,proposal_ms:f64,waiting_ms:f64,entry:u64,started:Instant,proposed:usize,waited:usize) {
        self.open=Some(Cycle {parts:[proposal_ms,waiting_ms,0.0,0.0],entry,started,proposed:proposed as u64,drafted:0,waited:waited as u64});
    }
    /// A pass of `rows` rows took `ms`. A pass of one row ends an open cycle with no verdict:
    /// every proposed row was cut before the pass, and the proposal is counted as unused.
    pub(super) fn pass(&mut self,rows:usize,ms:f64,picked:bool) {
        if rows>1 {if let Some(cycle)=self.open.as_mut() {cycle.parts[2]+=ms;cycle.drafted+=rows as u64-1;}}
        else {self.begin();self.steps+=1;self.picked+=u64::from(picked);}
    }
    /// The verdict of the open cycle: the rows are kept up to `kept_to`, in `ms`; `partial` when rows were dropped.
    pub(super) fn verdict(&mut self,kept_to:u64,ms:f64,partial:bool) {
        let Some(mut cycle)=self.open.take() else {return;};cycle.parts[3]=ms;let mut parts=cycle.parts;
        self.edge_ms+=(cycle.started.elapsed().as_secs_f64()*1000.0-parts.iter().sum::<f64>()).max(0.0);
        if partial {self.partial+=1;self.partial_ms+=parts[3];}
        if self.tried==0 {self.seed_ms=parts[1];parts[1]=0.0;}
        let ms=parts.iter().sum::<f64>();for (sum,part) in self.parts.iter_mut().zip(parts) {*sum+=part;}
        self.proposed+=cycle.proposed;self.drafted+=cycle.drafted;self.waited+=cycle.waited;
        self.kept+=kept_to.saturating_sub(cycle.entry) as f64;self.cycle_ms+=ms;self.tried+=1;
    }
    /// What a proposing cycle costs beside its pass and its proposals: the waiting rows and the verdict.
    pub(super) fn beside_ms(&self)->f64 {if self.tried==0 {0.0} else {(self.parts[1]+self.parts[3])/f64::from(self.tried)}}
    pub(super) fn account(&self)->Account {
        Account {cycles:self.tried,proposed_tokens:self.proposed,drafted_rows:self.drafted,kept_rows:self.kept as u64,proposal_ms:self.parts[0],
            unused_proposals:self.unused,unused_tokens:self.unused_tokens,unused_ms:self.unused_ms,one_row_passes:self.steps,picked_on_device:self.picked}
    }
    /// The request's two account lines. The first keeps the fields it had; the second adds the rest.
    pub(super) fn report(&self,rank:u32,session:u64,depth:usize,step_ms:f64) {
        let cycles=f64::from(self.tried.max(1));let mean=|part:usize|self.parts[part]/cycles;
        eprintln!("SERVED PROPOSER rank={rank} session={session} depth={depth} cycles_proposed={} cycles_declined={} kept_tokens_a_cycle={:.3} \
cycle_ms={:.1} step_ms={step_ms:.1} mean_proposal_ms={:.1} mean_waiting_rows_ms={:.1} mean_pass_ms={:.1} mean_verdict_ms={:.1} prompt_rows_ms={:.1}",
            self.tried,self.declined,self.kept/cycles,self.cycle_ms/cycles,mean(0),mean(1),mean(2),mean(3),self.seed_ms);
        eprintln!("SERVED PROPOSER PARTS rank={rank} session={session} proposed_tokens={} waiting_rows={} mean_edge_ms={:.2} \
partial_verdicts={} mean_partial_verdict_ms={:.2} one_row_passes={} picked_on_device={} drafted_rows={} \
unused_proposals={} unused_proposed_tokens={} unused_proposal_ms={:.2}",
            self.proposed,self.waited,self.edge_ms/cycles,self.partial,self.partial_ms/f64::from(self.partial.max(1)),self.steps,self.picked,
            self.drafted,self.unused,self.unused_tokens,self.unused_ms);
    }
}
