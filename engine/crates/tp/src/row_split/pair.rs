//! PAIR DRAFTING: two strict chats that are alone on the server draft in turns.
//!
//! Until now a proposal was verified for one chat, and chats that ran together decoded plain rows. Two chats can
//! each keep their own proposer and their own check pass: nothing of a pass is shared, so the passes simply follow
//! each other in one tick of the scheduler ("turns"). Each row of each pass still has the bits of that chat's plain
//! step, and each chat keeps only the tokens its own rows chose.
//!
//! WHAT IT COSTS (two Sparks, main `1508efce96`, 2026-11-15, 256 tokens a chat): one step of two chats is 66.1 ms
//! for two tokens, 30.2 tok/s together. A cycle of one chat at depth 3 is 105 ms, so a tick in turns is about 210 ms:
//! it pays when each chat keeps more than 3.1 rows of its 4. Structured chats keep 4 of 4 after their opening.
//!
//! So the PAIR decides, not each chat: a chat that drafted while the other took plain steps would get its tokens at
//! the other's cost (one token in 145 ms against one in 66 ms). Both chats propose their whole depth in a tick, or
//! neither does and both ride the step of two chats as before.
//!
//! Each chat first runs a lookback trial on shared plain steps (trial.rs PAIR TRIAL: it starts eight steps into the
//! answer, and counts the tokens its probes matched). Both must pass before any speculative check runs. A
//! failing chat ends the trial for the pair, so its partner cannot make it pay for unprofitable turns. Shared steps
//! retain session-local predictor inputs during this trial. After real turns start, leaving them is final. The price
//! uses measured shared steps when available.
//!
//! SHARED PASS (shared.rs, 2026-11-16). A tick "in turns" no longer runs a pass a chat: the rows of both chats are
//! ONE pass, in which every weight is read once for both. The rule below did not change: it reads what each tick
//! cost, and a shared tick costs less, so the record pays at fewer kept rows than the 3.1 of two passes.
//!
//! THE RULE. Each chat's first PAIR_OPEN cycles are in turns and are not in the record: answers open with a few
//! cycles that lose rows whatever follows (structured kept 2, 1, 2 and then 4 for the rest; the same chats alone do
//! the same). A chat that joins a chat that drafts already has its own opening. When both chats have had theirs the
//! record starts as PAIR_TRUST cycles that kept everything, takes every later cycle (rows each chat kept, ms of
//! both), and forgets like price.rs's. The pair stays in turns while the chat that kept FEWER rows on
//! the record still gets its tokens at PAIR_FLOOR of the plain rate or more: neither chat is slower than in the
//! step of two.
//!
//! GROUPS OF THREE (2026-11-16). The shared pass holds the rows of up to PAIR_CHATS chats (12 rows at depth
//! 3), so the same rule serves them: every chat that is live is a strict chat of the group, every chat's trial must
//! pass, all propose their whole depth in a tick or none does, and the group stays while the chat that kept the
//! FEWEST rows still gets its tokens at PAIR_FLOOR of the rate of the step of all of them. A chat that comes or goes
//! starts a new trial and record. When a chat ends after every trial passed, the remaining chats keep their trial
//! verdicts and start a new record. "Pair" below reads as "the chats that draft together".
//!
//! Nothing here can change a kept token: every proposed row is still verified by its own chat's pass.
/// The share of two chats' own plain steps that one step of the two takes. MEASURED, two Sparks, main
/// `1508efce96`, 2026-11-15: 66.1 ms against two own steps of 40.0 ms. Bits never depend on it.
pub(super) const PAIR_STEP_SHARE:f64=0.83;
/// The most strict chats that draft together. Three gained in the balanced duox-k bracket; four
/// varied between a gain and a loss and stays plain until its check cost improves. The row bound stays unchanged.
pub(super) const PAIR_CHATS:usize=3;
const _:()=assert!(PAIR_CHATS*(super::trial::TRIAL_DEPTH+1)<=crate::bounded_decoder::VERIFY_ROWS && PAIR_CHATS<=crate::bounded_decoder::BATCH_ROWS);
/// The share of the chats' own plain steps that one step of all of them takes, before that step was timed. Three
/// and four chats: MEASURED 2026-11-12 (batch.rs BATCH_MIN_ROWS), 95.3 and 109.3 ms against own steps of 42.6 ms.
fn step_share(chats:usize)->f64 {match chats {0..=2=>PAIR_STEP_SHARE,3=>0.75,_=>0.64}}
/// Turns must make tokens at this share of the plain rate or more. Above 1: the plain price is an estimate from each
/// chat's first timed step, which is slow (43 ms against 40.0 steady: the estimate reads up to 8 percent high).
pub(super) const PAIR_FLOOR:f64=1.05;
/// Ticks in turns in a row in which no chat could propose, before the pair gives up: a chat's first proposal needs
/// a timed plain step of its own (two passes of one row). Also the ticks ONE chat may go without a proposal beside
/// a chat that proposes, before its passes of one row count as its cycles (it cannot wait at the other's cost).
pub(super) const PAIR_DRY:u32=2;
/// Each chat's proposing cycles at its start that are not in the record.
pub(super) const PAIR_OPEN:u32=3;
/// Prior whole cycles for two chats. Wider groups scale this by their chat count: the minimum of more records is
/// more sensitive to one early cost spike. The measured continuation floor stays the same.
pub(super) const PAIR_TRUST:f64=2.0;
/// How the record forgets (price.rs's DECAY): about the last 32 cycles.
const DECAY:f64=31.0/32.0;
/// What a pair counted, for the served log, the check tools and the tests.
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct PairAccount {
    /// Ticks in turns that reached both chats' verdicts, and those of them in which no chat proposed.
    pub turn_ticks:u32,pub dry_ticks:u32,
    /// Steps of the two chats the pair rode.
    pub plain_steps:u32,
    /// Rows each chat's turns verified beyond its own next token, and rows each chat's turns kept.
    pub drafted_rows:[u64;PAIR_CHATS],pub kept_rows:[u64;PAIR_CHATS],
    /// Sum of both chats' cycles over the ticks in turns, ms.
    pub turn_ms:f64,
    /// The pair's timed step of two chats, ms. None: never timed.
    pub step_ms:Option<f64>,
    /// Proposing cycles; whether the pair left its turns; the record when it ended (rows each chat kept, ms of
    /// both); and the plain price the last decision used, ms.
    pub cycles:u32,pub left:bool,pub record_rows:[f64;PAIR_CHATS],pub record_ms:f64,pub price_ms:f64}
/// The strict chats that can draft beside another chat, and the record of the two of them that are a pair now.
#[derive(Default)]
pub(super) struct Pair {
    /// Strict chats of depth 2 or more with a proposer, in the order they came.
    chats:Vec<u64>,
    /// Each chat's own timed plain step, ms, and its whole depth, as it last said.
    own:[f64;PAIR_CHATS],whole:[usize;PAIR_CHATS],
    /// The timed step of the chats together, ms (steps that follow such a step), and whether the last call was one.
    step:Option<f64>,after_step:bool,
    /// The record: rows each chat kept and the ms of both chats' cycles, decayed. Empty during the opening.
    kept:[f64;PAIR_CHATS],ms:f64,
    /// The pair left its turns. It does not come back: its predictor caches are gone.
    over:bool,
    /// Per-chat final stop decisions survive another chat's close until this chat closes too.
    stopped:Vec<u64>,
    /// The plain price of the last decision, ms.
    price:f64,
    /// This tick: a chat asked; what both are told; the step ran in turns; what each chat's turn drafted, kept and cost.
    asked:bool,turns:bool,running:bool,turn:[Option<(u64,u64,f64)>;PAIR_CHATS],
    /// Ticks in turns that drafted rows, and ticks in turns in a row that drafted none.
    cycles:u32,dry:u32,
    /// Each chat's own cycles, and its ticks in a row without a proposal beside a chat that proposed.
    drafts:[u32;PAIR_CHATS],idle:[u32;PAIR_CHATS],
    /// Test control: every tick of a pair is in turns.
    always:bool,
    trial:[super::Trial;PAIR_CHATS],ready:[bool;PAIR_CHATS],
    account:PairAccount}
impl Pair {
    /// A strict chat that can draft beside another opened its request.
    pub(super) fn join(&mut self,id:u64) {if !self.chats.contains(&id) {self.chats.push(id);self.fresh();}}
    /// It ended or gave its drafting up. The account of the pair it was in, if it was in one that ran a tick.
    pub(super) fn leave(&mut self,id:u64)->Option<(Vec<u64>,PairAccount)> {
        let at=self.chats.iter().position(|&chat|chat==id)?;
        let ended=(self.group()>0 && self.account!=PairAccount::default()).then(|| (self.chats.clone(),
            PairAccount {step_ms:self.step,cycles:self.cycles,left:self.over,record_rows:self.kept,record_ms:self.ms,price_ms:self.price,..self.account}));
        // Three or four chats that were all in shared passes: the ones that are left go on in them. Their trials
        // stand (each read that chat's own text); the record and the price start again, for the chats there are now.
        let stays=self.group()>2 && !self.over && !self.always && self.ready[..self.chats.len()].iter().all(|&ready|ready);
        self.chats.remove(at);self.stopped.retain(|&chat|chat!=id);self.fresh();
        if stays {self.ready=[true;PAIR_CHATS];}
        ended
    }
    /// A new set of chats starts a new record: nothing of another pair prices this one.
    fn fresh(&mut self) {
        let (chats,stopped,always)=(std::mem::take(&mut self.chats),std::mem::take(&mut self.stopped),self.always);
        let over=chats.iter().any(|id|stopped.contains(id));*self=Self {chats,stopped,always,trial:std::array::from_fn(|_|super::Trial::pair()),..Self::default()};
        if over {self.stop();}
    }
    fn stop(&mut self) {self.over=true;if !self.always {for &id in &self.chats {if !self.stopped.contains(&id) {self.stopped.push(id);}}}}
    pub(super) fn set_always(&mut self,always:bool) {self.always=always;}
    /// Whether an answer of `output_cap` tokens can pay a pair trial back. The earliest first shared tick is the step
    /// after the last of the fewest probes that can start (PAIR TRIAL, trial.rs: PAIR_SKIP steps, then probes
    /// PAIR_GAP apart until PAIR_NEED tokens matched). The cap must hold the PAIR_OPEN opening ticks after it: the
    /// probes of both chats cost 92 ms and a shared tick saves about 86 ms against four steps of two chats (two
    /// Sparks, 2026-11-16, run duox-s1). A shorter answer uses ordinary shared decoding with no pair state.
    pub(super) fn can_trial(&self,output_cap:usize)->bool {self.always || output_cap>=Self::trial_room()}
    pub(super) fn trial_room()->usize {
        use super::trial::{PAIR_GAP,PAIR_NEED,PAIR_SKIP,TRIAL_DEPTH};
        let probes=PAIR_NEED.div_ceil(TRIAL_DEPTH as u32);
        let first_turn=PAIR_SKIP+1+(probes-1)*(PAIR_GAP+1)+1;
        (first_turn+PAIR_OPEN*(TRIAL_DEPTH as u32+1)) as usize
    }
    /// The chats that draft together: 2 to PAIR_CHATS strict chats. 0: one chat, or more than a shared pass holds.
    pub(super) fn group(&self)->usize {if (2..=PAIR_CHATS).contains(&self.chats.len()) {self.chats.len()} else {0}}
    /// Whether three chats may draft in this tick: the scheduler then gives a tick the rows of a whole pass.
    pub(super) fn wide(&self)->bool {self.group()>2 && !self.over}
    /// Chat `id`'s place in the pair, when the pair is all the chats that are live (`live` sessions are open).
    fn place(&self,id:u64,live:usize)->Option<usize> {if self.group()>0 && live==self.chats.len() {self.chats.iter().position(|&chat|chat==id)} else {None}}
    /// Both chats hand their pending canonical token back when this pair gives drafting up.
    /// A chat that closes after its handoff cannot erase the survivor's verdict.
    pub(super) fn stopped(&self,id:u64)->bool {!self.always && self.stopped.contains(&id)}
    /// Chat `id`, whose whole depth is `whole` and whose own plain step takes `own` ms, is about to propose beside
    /// another chat: whether it proposes its whole depth in this tick. The first ask of a tick decides for both.
    pub(super) fn ask(&mut self,id:u64,live:usize,whole:usize,own:Option<f64>)->bool {
        let Some(at)=self.place(id,live) else {return false;};
        if let Some(ms)=own {self.own[at]=ms;}
        self.whole[at]=whole;
        if !self.asked {self.asked=true;self.turn=[None;PAIR_CHATS];self.turns=self.decide(whole);}
        self.turns
    }
    /// Keep plain-step predictor inputs only while both chats are still eligible for the trial.
    pub(super) fn tracking(&self,ids:&[u64])->bool {
        !self.always && !self.over && self.group()>0 && ids.len()==self.chats.len() && ids.iter().all(|id|self.chats.contains(id)) && !self.ready[..self.chats.len()].iter().all(|&ready|ready)
    }
    pub(super) fn probe_due(&mut self,id:u64,token:u32,marker:Option<u32>)->bool {
        let Some(at)=self.place(id,self.chats.len()) else {return false;};
        !self.over && !self.ready[at] && self.trial[at].due(token,marker)
    }
    pub(super) fn probed(&mut self,id:u64,matched:usize,ms:f64,waiting:f64) {
        let Some(at)=self.place(id,self.chats.len()) else {return;};
        let verdict=self.trial[at].record(super::trial::TRIAL_DEPTH,matched,ms,waiting);
        self.ready[at]=verdict==super::trial::Verdict::Start;
        if verdict==super::trial::Verdict::GiveUp {self.stop();}
        eprintln!("SERVED PAIR TRIAL session={id} verdict={verdict:?} matched={} proposal_ms={:.3} seed_ms={:.3}",self.trial[at].matched(),self.trial[at].proposal_ms(),self.trial[at].seed_ms());
    }
    fn decide(&mut self,whole:usize)->bool {
        if whole<2 {return false;}
        if self.always {return true;}
        let chats=self.chats.len();
        if self.over || !self.ready[..chats].iter().all(|&ready|ready) {return false;}
        // No chat got to a proposal: the pair cannot draft here.
        if self.dry>PAIR_DRY {self.stop();return false;}
        // The opening: the chats warm up, and their first cycles are not judged.
        if self.opened()<PAIR_OPEN {return true;}
        self.price=self.step.unwrap_or(step_share(chats)*self.own[..chats].iter().sum::<f64>());
        let pays=self.kept[..chats].iter().copied().fold(f64::INFINITY,f64::min)*self.price>=PAIR_FLOOR*self.ms;
        if !pays {self.stop();}
        pays
    }
    /// The cycles of the chat that has had fewer: the opening is over when both have had PAIR_OPEN.
    fn opened(&self)->u32 {self.drafts[..self.chats.len().min(PAIR_CHATS)].iter().copied().min().unwrap_or(0)}
    /// The scheduler's step of this tick starts: whether the pair is in turns in it (its chats then keep their own passes).
    pub(super) fn step_begins(&mut self)->bool {let turns=self.asked && self.turns;self.asked=false;self.running=turns;if !turns {self.turn=[None;PAIR_CHATS];}turns}
    /// Chat `id`'s turn of this tick verified `drafted` proposed rows and kept `kept` rows, its own row included, in `ms`.
    pub(super) fn turned(&mut self,id:u64,live:usize,drafted:u64,kept:u64,ms:f64) {
        if !self.running {return;}let Some(at)=self.place(id,live) else {return;};
        self.turn[at]=Some((drafted,kept,ms));let chats=self.chats.len();
        // The tick ends when every chat's turn is in.
        if self.turn[..chats].iter().any(Option::is_none) {return;}
        let turns=self.turn[..chats].iter().map(|turn|turn.unwrap_or_default()).collect::<Vec<_>>();
        self.turn=[None;PAIR_CHATS];self.running=false;self.after_step=false;
        let ms=turns.iter().map(|turn|turn.2).sum::<f64>();
        let account=&mut self.account;account.turn_ticks+=1;account.turn_ms+=ms;
        for (at,turn) in turns.iter().enumerate() {account.drafted_rows[at]+=turn.0;account.kept_rows[at]+=turn.1;}
        // Warming up is not a record of what proposals keep.
        if turns.iter().all(|turn|turn.0==0) {self.dry+=1;account.dry_ticks+=1;return;}
        self.dry=0;self.cycles+=1;let opened=self.opened();
        for (at,turn) in turns.iter().enumerate() {
            // A chat that proposes nothing beside one that does has PAIR_DRY ticks to warm up. After them its pass
            // of one row is its cycle: it kept one row.
            if turn.0>0 {self.drafts[at]+=1;self.idle[at]=0;} else {self.idle[at]+=1;if self.idle[at]>PAIR_DRY {self.drafts[at]+=1;}}
        }
        if self.opened()<PAIR_OPEN {return;}
        if opened<PAIR_OPEN {
            // The record starts with a whole-pass prior scaled to the number of independent chats.
            let trust=PAIR_TRUST*chats as f64/2.0;
            for at in 0..chats {self.kept[at]=trust*(self.whole[at]+1) as f64;}self.ms=trust*ms;return;
        }
        for (at,turn) in turns.iter().enumerate() {self.kept[at]=self.kept[at]*DECAY+turn.1 as f64;}
        self.ms=self.ms*DECAY+ms;
    }
    /// The chats of the pair rode one step of all of them in `ms`. After a tick in turns that ends the turns for good.
    pub(super) fn stepped(&mut self,ids:&[u64],ms:f64) {
        if self.group()==0 || ids.len()!=self.chats.len() || !ids.iter().all(|id|self.chats.contains(id)) {return;}
        if self.account.turn_ticks>0 {self.stop();}
        self.account.plain_steps+=1;self.dry=0;
        // The first step after other work is slower: only a step that follows a step is timed.
        if self.after_step {self.step=Some(self.step.map_or(ms,|old|0.75*old+0.25*ms));}
        self.after_step=true;
    }
}
impl<C:super::Compute,E:super::Exchange> super::Factory<'_,C,E> {
    /// A chat leaves the pair record (its request ended or gave its drafting up). A pair that ran prints its account.
    #[cold] #[inline(never)]
    pub(super) fn pair_leave(&self,id:u64) {
        let Some((ids,account))=self.pair.borrow_mut().leave(id) else {return;};
        let mut counts=self.pair_counts.get();counts[0]+=1;counts[1]+=account.turn_ticks;counts[2]+=account.plain_steps;self.pair_counts.set(counts);
        eprintln!("SERVED PAIR rank={} sessions={ids:?} turn_ticks={} dry_ticks={} cycles={} plain_steps={} left_turns={} drafted_rows={:?} kept_rows={:?} turn_ms={:.1} step_ms={:.1} step_timed={} record_rows=[{}] record_ms={:.1} price_ms={:.1}",
            self.split.topology.rank,account.turn_ticks,account.dry_ticks,account.cycles,account.plain_steps,u8::from(account.left),&account.drafted_rows[..ids.len()],&account.kept_rows[..ids.len()],account.turn_ms,account.step_ms.unwrap_or(0.0),u8::from(account.step_ms.is_some()),
            account.record_rows[..ids.len()].iter().map(|rows|format!("{rows:.2}")).collect::<Vec<_>>().join(", "),account.record_ms,account.price_ms);
    }
}
#[cfg(test)] mod tests {
    use super::*;
    /// One tick of a pair of depth 3: both ask, the step begins, and (in turns) both verdicts come. `kept` 0 is a
    /// turn that proposed nothing.
    fn tick(pair:&mut Pair,kept:[u64;2],ms:f64)->bool {
        let turns=pair.ask(7,2,3,Some(40.0));assert_eq!(pair.ask(9,2,3,Some(40.0)),turns,"both chats are told the same in one tick");
        assert_eq!(pair.step_begins(),turns);
        if turns {pair.turned(7,2,if kept[0]>0 {3} else {0},kept[0].max(1),ms);pair.turned(9,2,if kept[1]>0 {3} else {0},kept[1].max(1),ms);}
        else {pair.stepped(&[7,9],66.0);}
        turns
    }
    fn pair()->Pair {let mut pair=Pair::default();pair.join(7);pair.join(9);pair.ready=[true;PAIR_CHATS];pair}
    /// The ticks in turns of a pair whose cycles keep `rows(i)`, of `ticks` ticks after two warm-up ticks.
    fn run(ticks:usize,rows:impl Fn(usize)->[u64;2])->(u32,PairAccount) {
        let mut pair=pair();assert!(tick(&mut pair,[0,0],40.0) && tick(&mut pair,[0,0],40.0));
        let mut turns=0;for at in 0..ticks {turns+=u32::from(tick(&mut pair,rows(at),105.0));}
        (turns,pair.leave(7).unwrap().1)
    }
    #[test] fn both_trials_must_pay_before_turns_and_one_failure_keeps_both_plain() {
        let mut p=Pair::default();p.join(7);p.join(9);
        assert!(p.tracking(&[7,9]));assert!(!tick(&mut p,[4,4],105.0));
        for _ in 0..3 {p.probed(7,3,10.0,0.0);}
        assert!(!tick(&mut p,[4,4],105.0),"one paying chat cannot charge its partner");
        // PAIR TRIAL (trial.rs): one probe that matched nothing does not end it; the second puts 7 tokens out of reach.
        p.probed(9,0,4.0,0.0);assert!(p.tracking(&[7,9]) && !p.stopped(9));
        p.probed(9,0,4.0,0.0);
        assert!(!p.tracking(&[7,9]));assert!(!tick(&mut p,[4,4],105.0));
        assert!(p.stopped(7) && p.stopped(9));assert!(!p.stopped(8));
        p.leave(7);assert!(!p.stopped(7) && p.stopped(9));
        p.join(11);assert!(p.stopped(11));p.leave(9);p.leave(11);assert!(p.stopped.is_empty());
        p.set_always(true);assert!(!p.stopped(7));
        let mut p=Pair::default();p.join(7);p.join(9);
        for _ in 0..3 {for id in [7,9] {p.probed(id,3,10.0,0.0);}}
        assert!(!p.tracking(&[7,9]));assert!(tick(&mut p,[4,4],105.0));
    }
    #[test] fn a_pair_is_two_strict_chats_alone_on_the_server() {
        let mut one=Pair::default();one.join(7);
        assert!(!one.ask(7,1,3,Some(40.0)),"one chat is not a pair");assert!(!one.ask(7,2,3,Some(40.0)),"the other live chat is not a drafting chat");
        let mut pair=pair();
        assert!(!pair.ask(7,3,3,Some(40.0)),"a third live chat: plain rows, as before");assert!(!pair.ask(8,2,3,Some(40.0)),"a chat that is not of the pair");
        assert!(!pair.ask(7,2,1,Some(40.0)),"a pair of depth 1 never drafts: its pass of two rows costs more than two steps");
        assert!(!pair.step_begins());
        assert!(pair.ask(7,2,3,Some(40.0)) && pair.ask(9,2,3,None),"the first ask of a tick decides for both");assert!(pair.step_begins());
        pair.turned(7,2,3,4,105.0);pair.turned(9,2,3,4,105.0);
        assert!(!pair.step_begins(),"a step with no ask before it is not in turns");
        // A verdict outside a tick in turns is not counted. A chat that leaves ends the pair, and the next pair starts a new record.
        pair.turned(7,2,3,4,105.0);pair.turned(9,2,3,4,105.0);
        let (ids,account)=pair.leave(9).unwrap();assert_eq!(ids,[7,9]);assert_eq!((account.turn_ticks,account.kept_rows,account.drafted_rows,account.cycles),(1,[4,4,0],[3,3,0],1));
        assert!(!pair.ask(7,2,3,Some(40.0)));pair.join(11);assert!(pair.leave(11).is_none(),"a pair that ran no tick prints nothing");
    }
    #[test] fn turns_go_on_only_while_neither_chat_is_slower_than_in_the_step_of_two() {
        // Structured text (kept 4 of 4 by both, 105 ms a cycle): 4 rows a tick each for 210 ms, against 66 ms a row.
        let (turns,account)=run(40,|_|[4,4]);
        assert_eq!((turns,account.turn_ticks,account.dry_ticks,account.plain_steps,account.left),(40,42,2,0,false));assert_eq!(account.step_ms,None);
        // Code (3 of 4 kept by both): 3 rows a tick each for 210 ms loses against 3 steps of 66 ms. The pair leaves
        // when its record says so, and does not come back.
        let (turns,account)=run(200,|_|[3,3]);
        assert!((PAIR_OPEN+1..12).contains(&turns),"{turns} ticks of 200 were in turns");assert!(account.left && account.plain_steps>=188);
        assert!((account.step_ms.unwrap()-66.0).abs()<1e-9 && (account.price_ms-66.4).abs()<1e-9,"{account:?}");
        // One chat keeps everything and the other only its own row: the pair leaves the turns, because the second
        // chat would get one token in 210 ms against one in 66 ms.
        let (turns,account)=run(200,|_|[4,1]);assert!((PAIR_OPEN+1..8).contains(&turns) && account.left,"{turns} ticks of 200 were in turns");
        // Chats that never get to a proposal leave the turns after PAIR_DRY ticks and one more.
        let mut dry=pair();let mut turns=0;
        for _ in 0..40 {turns+=u32::from(tick(&mut dry,[0,0],40.0));}
        assert_eq!(turns,PAIR_DRY+1);
    }
    #[test] fn the_opening_is_not_judged_and_leaving_is_final() {
        // MEASURED 2026-11-15 (two Sparks): what two structured chats kept in their first cycles together, then 4 of 4.
        let (first,second)=([2,4,2,4,4,3,4,4,2,3],[2,1,2,4,4,4,4,4,4,4]);
        let (turns,account)=run(120,|at|[first.get(at).copied().unwrap_or(4),second.get(at).copied().unwrap_or(4)]);
        assert_eq!((turns,account.left,account.plain_steps),(120,false,0),"the opening (2, 1, 2 rows) does not end a pair that keeps its rows after it");
        // The same opening, then prose (2 rows of 4): out within a few cycles.
        let (turns,account)=run(120,|at|if at<3 {[first[at],second[at]]} else {[2,2]});
        assert!((PAIR_OPEN+1..=PAIR_OPEN+4).contains(&turns) && account.left,"{turns}");
        // A pair that left stays out, even when every proposal would be kept from then on: its predictor caches are
        // gone, and a try after a step of two measures a cold predictor.
        let mut out=pair();tick(&mut out,[0,0],40.0);tick(&mut out,[0,0],40.0);
        let mut at=0;while tick(&mut out,[1,1],105.0) {at+=1;assert!(at<20);}
        for _ in 0..300 {assert!(!tick(&mut out,[4,4],105.0));}
        // One bad run does not end a pair with a long good record: nine cycles that keep 2 rows after forty whole ones.
        let (turns,_)=run(49,|at|if at<40 {[4,4]} else {[2,2]});assert_eq!(turns,49);
        // A step of two after a tick in turns ends the turns too (the open rows were settled).
        let mut settled=pair();assert!(tick(&mut settled,[0,0],40.0));settled.stepped(&[7,9],66.0);assert!(!tick(&mut settled,[4,4],105.0));
        // Before any tick in turns it ends nothing: no open rows existed.
        let mut early=pair();early.stepped(&[7,9],66.0);assert!(tick(&mut early,[0,0],40.0));
    }
    #[test] fn a_chat_that_joins_a_drafting_chat_has_its_own_opening_and_cannot_wait_for_ever() {
        // Chat 7 drafts already and keeps 4 of 4. Chat 9 joins: two ticks to warm up, an opening that loses rows
        // (2, 1, 2), then 4 of 4. Its opening is not in the record, so the pair stays.
        let late=[0,0,2,1,2];let mut joined=pair();
        for at in 0..80 {assert!(tick(&mut joined,[4,late.get(at).copied().unwrap_or(4)],105.0),"tick {at}");}
        assert!(!joined.leave(7).unwrap().1.left);
        // A chat that never proposes beside one that does: after PAIR_DRY ticks its passes of one row are its
        // cycles, the opening ends, and the record ends the turns. It does not wait at the other chat's cost.
        let mut waiting=pair();let mut turns=0;
        for _ in 0..80 {turns+=u32::from(tick(&mut waiting,[4,0],105.0));}
        assert!((PAIR_DRY+PAIR_OPEN..=PAIR_DRY+PAIR_OPEN+2).contains(&turns),"{turns}");assert!(waiting.leave(7).unwrap().1.left);
    }
    #[test] fn the_test_control_keeps_every_tick_in_turns() {
        let mut pair=pair();pair.set_always(true);for _ in 0..30 {assert!(tick(&mut pair,[1,1],105.0));}
        pair.stepped(&[7,9],66.0);assert!(tick(&mut pair,[1,1],105.0),"with the control a step of two ends nothing");
        pair.join(11);assert!(pair.ask(7,3,3,None),"three strict chats draft together, with the control from their first tick");
        pair.join(12);pair.join(13);assert!(!pair.ask(7,5,3,None),"five chats are more than a shared pass holds: plain rows");
    }
    /// One tick of `chats` strict chats of depth 3 (ids 1, 2, ...): all ask, the step begins, and (in turns) every verdict
    /// comes with the rows that chat kept. `ms`: each chat's share of the tick. A plain tick is a step of all in `step` ms.
    fn group_tick(pair:&mut Pair,kept:&[u64],ms:f64,step:f64)->bool {
        let (chats,ids)=(kept.len(),(1..=kept.len() as u64).collect::<Vec<_>>());
        let turns=pair.ask(1,chats,3,Some(40.0));for &id in &ids[1..] {assert_eq!(pair.ask(id,chats,3,Some(40.0)),turns,"every chat is told the same in one tick");}
        assert_eq!(pair.step_begins(),turns);
        if turns {for (&id,&kept) in ids.iter().zip(kept) {pair.turned(id,chats,if kept>0 {3} else {0},kept.max(1),ms);}} else {pair.stepped(&ids,step);}
        turns
    }
    #[test] fn three_strict_chats_draft_together_while_the_chat_that_keeps_fewest_still_gains() {
        let group=|| {let mut pair=Pair::default();for id in 1..=3 {pair.join(id);}pair};
        let mut three=group();assert_eq!(three.group(),3);assert!(three.wide());
        for id in 1..=2 {for _ in 0..3 {three.probed(id,3,10.0,0.0);}}
        assert!(three.tracking(&[1,2,3]));
        assert!(!group_tick(&mut three,&[4;3],80.0,90.0),"the third chat must pass its own trial");
        for _ in 0..3 {three.probed(3,3,10.0,0.0);}
        assert!(!three.tracking(&[1,2,3]));
        for at in 0..60 {assert!(group_tick(&mut three,&[4;3],80.0,90.0),"tick {at}");}
        let mut turns=0;for _ in 0..60 {turns+=u32::from(group_tick(&mut three,&[4,4,1],80.0,90.0));}
        assert!((1..20).contains(&turns),"the weakest chat must leave before another twenty cycles: {turns}");
        let (ids,account)=three.leave(2).unwrap();assert_eq!(ids,[1,2,3]);assert!(account.left && account.kept_rows[2]<account.kept_rows[0]);
        assert_eq!(three.group(),2);assert!(three.stopped(1));
        // Three paying chats become two without repeating the survivors' passed trials.
        let mut three=group();for id in 1..=3 {for _ in 0..3 {three.probed(id,3,10.0,0.0);}}
        for _ in 0..20 {assert!(group_tick(&mut three,&[4;3],80.0,90.0));}
        assert!(!three.leave(3).unwrap().1.left);assert_eq!(three.group(),2);assert!(!three.tracking(&[1,2]));
        assert!(three.ask(1,2,3,Some(40.0)) && three.ask(2,2,3,Some(40.0)) && three.step_begins());
        three.turned(1,2,3,4,80.0);three.turned(2,2,3,4,80.0);three.leave(2);assert_eq!(three.group(),0);
        // One failed trial, or another live chat outside the group, keeps the group plain.
        let mut failed=group();for id in 1..=2 {for _ in 0..3 {failed.probed(id,3,10.0,0.0);}}
        failed.probed(3,0,10.0,0.0);failed.probed(3,0,10.0,0.0);
        assert!(failed.stopped(1) && failed.stopped(2) && failed.stopped(3));
        let mut mixed=group();for id in 1..=3 {for _ in 0..3 {mixed.probed(id,3,10.0,0.0);}}
        assert!(!mixed.ask(1,4,3,Some(40.0)));
    }
    #[test] fn a_fourth_chat_keeps_the_group_on_plain_rows() {
        assert_eq!(PAIR_CHATS,3);
        let mut four=Pair::default();for id in 1..=4 {four.join(id);}
        assert_eq!(four.group(),0);assert!(!four.wide());assert!(!four.tracking(&[1,2,3,4]));
        for id in 1..=4 {assert!(!four.ask(id,4,3,Some(40.0)));}
        assert!(!four.step_begins());
        assert_eq!(four.account.cycles,0);
    }

}
