//! GROUP DRAFTING: chats that share the server draft together, in ONE shared pass, and no chat pays for another's.
//!
//! Until 2026-11-17 two or three strict chats drafted all together or not at all ("pair drafting"): a lookback
//! trial of each chat had to pass before any of them proposed, then all proposed their whole depth in every tick,
//! and the group left for good when the chat that kept fewest rows lost. A request that named no draft mode decoded
//! plain rows beside another chat.
//!
//! MEASURED 2026-11-17 (two Sparks, main `d98c9866e6`, run draftm-base-m-1, 256 tokens a chat): that rule kept two
//! strict structured chats plain for their whole answers (31.30 tok/s against 31.37 plain): their trial gave up at
//! its first two probes, inside the think block. Two strict prose chats were plain too. Two strict code chats
//! drafted: 35.69 tok/s against 31.23.
//!
//! MEASURED the same day (run draftm-cell-n-1): a rule that let each chat draft by the gain of the GROUP's tokens
//! raised two structured chats from 31.47 to 40.20 tok/s. It also made a chat slower whenever its neighbour drafted
//! and it did not: of two prose chats one ran 24.4 percent under its plain rate (and the two together 7.5 percent
//! under), a prose chat beside a structured one 5.4 percent under. On a shared server the slow chat belongs to
//! another person. So the rule here is FAIR: rows are admitted only when every chat of the group gains.
//!
//! MEASURED the same day (run draftm-fair-f-1): a FAIR rule that tried one word of every chat at growing gaps and
//! gave up after six tries gained nothing (two structured chats 31.44 to 31.17 tok/s). Each try cost every chat a
//! fifth of a token, and every group gave up before tick 40: the answer text of a structured chat starts near tick
//! 90. A step that keeps the proposers ready costs what a plain step costs (63.0 to 64.1 ms against 62.8 to 63.8),
//! so a group that waits loses nothing. Hence a look is one chat's, it is paid from a small allowance, and the
//! group never gives up.
//!
//! MEASURED the same day (run draftm-look-g-1): with such looks two structured chats rose from 31.43 to 35.86
//! tok/s, each chat 14 percent over its plain rate. Two faults showed. A look that passed by its price alone began
//! a drafting that kept 1 of 4 checked words (two prose chats, 2.5 tokens lost each): so a look must also be RIGHT.
//! And a chat's first proposal is dear and grows with the wait (15.5 to 17 ms at tick 39, 26 to 28 ms at tick 120:
//! the draft head starts from the prompt rows and every row decoded since; a later proposal costs 5 to 13 ms): so
//! a chat's first proposal is not made before it is worth a look.
//!
//! MEASURED the same day (run draftm-right-h-1): with looks that must be right no prose pair began to draft, and
//! neither did the two structured chats (8 looks, 31.42 to 31.14 tok/s). Their think blocks end at tokens 89 and
//! 114 of 256, so most looks fell inside a think block; and a drafting that did begin (two strict structured
//! chats) ended after one tick that kept 3 and 2 rows of 4, because it was judged against the looks before it. So
//! a chat is looked at only in its answer text, and a drafting is judged from its own start.
//!
//! WHAT IS TIMED. Everything is priced from this group's own clock.
//! (1) THE PLAIN PERIOD: the wall time of one tick in which the chats ride one batch step and nobody proposes. The
//! first PLAIN_SKIP ticks after a prompt are slow in plain decoding too (84 and 72 ms where the steady step is 63.4,
//! same run), so they are not timed. A chat's plain rate beside these chats is one token a plain period.
//! (2) THE PASS by its rows: the timed shared passes of the group, joined by straight lines (`rows_ms`).
//! (3) THE TIME OUTSIDE THE PASS of a tick that drafts: the proposals of all chats, their verdicts, the scheduler.
//! The first PASS_SKIP ticks of a drafting are slow too and are not its price (MEASURED, runs draftm-text-j-1 and
//! draftm-wait-k-1: passes of 12 rows 240.9 and 201.3 ms where a steady one takes 204, and 57.6 ms outside them
//! where the steady time is 32). Priced from them, three sure chats were refused their third tick by 0.02 token.
//! THE NET counts those ticks in full.
//!
//! THE RULE. The chats that are a group (2 to PAIR_CHATS chats, each a strict request or one that drafts by
//! default, and no other chat live) draft together or not at all. The words a chat's rows check are chosen by the
//! draft head's probability of each word, through the service's calibration (confidence.rs).
//! (a) A LOOK. While the group does not draft, ONE chat proposes its whole depth and no row checks it. The look
//! passes when that chat's words would pay beside chats as sure as it is, AND the tokens the chat then decodes are
//! the words it proposed, every one. After a look that passes the next chat looks. When every chat's last look has
//! passed, the group DRAFTS. A chat is looked at only in its ANSWER TEXT (`Text`): the words of a think block say
//! nothing about the text after it.
//! (b) DRAFTING. Every chat proposes its whole depth in every tick. A set of rows is ADMITTED only when EVERY chat
//! of the group is expected to keep enough words in that tick to cover the whole time the tick takes longer than a
//! plain period (PAIR_FLOOR times it). A chat that has chosen in this tick counts with its choice; one that has
//! not yet, with the chance its first words had lately. So a chat whose words are unsure keeps the rows of a sure
//! chat out: no chat falls under its plain rate for another's gain. The tick in which a chat's words do not pass
//! is the last of the drafting: that chat must pass a look again, at once, and then the group drafts on. (If a
//! chat before it brought rows to that tick, it checks the words that leave it better off than riding on them.)
//! (c) THE NET, measured: in every tick that proposed or checked a row, each chat's tokens are counted against its
//! plain pace on the wall clock. A look is made only while it leaves no chat more than LOOK of the group's plain
//! tokens so far behind that pace: a group that never drafts spends about that share on looking. It is a pace and not a cap: a chat's first
//! look costs more than the allowance expects, and a look that follows a right look is not held back by it
//! (MEASURED 2026-11-17: 0.2 to 0.43 percent of the tokens of a 256-token request). A chat whose
//! look FAILED, by its price or by a wrong word, waits for its next one as long again as it has been in its answer
//! text without a look that passed, or since the group last drafted (prose stays prose: looks near token 38, 105,
//! 194). No other chat looks while it waits, since the group cannot draft without it, and it looks first. It must
//! then be right twice in a row: a drafting that starts and ends costs
//! every chat more than its ticks say (MEASURED, run draftm-text-j-1: the first pass and the four steps after the
//! last one are slow, about 1.5 tokens a chat). A drafting goes on while no chat has lost more than ROOM
//! tokens since it began, whatever the prices said.
//!
//! WHAT IT LEAVES OUT: a chat that drafts well beside one that does not stays plain. The group's tokens together
//! would rise (structured beside prose: 30.92 to 34.83 tok/s, run draftm-base-m-1) and the other chat would wait.
//!
//! A chat that comes or goes starts a new record for the chats there are now: the step of a group of another size
//! has another price. Nothing here can change a kept token: every proposed row is verified by its own chat's rows.
use crate::bounded_decoder::VERIFY_ROWS;
use std::collections::BTreeMap;
/// The most chats that draft together. Widened to BATCH_ROWS (8) with grouped NVFP4 TC (`fp4_cohort_t8`) and 32-row verify.
pub(super) const PAIR_CHATS:usize=crate::bounded_decoder::BATCH_ROWS;
/// The groups whose chats propose in ONE command (one_pass.rs) and price the rows a pass computes, carried rows included:
/// two chats and four through PAIR_CHATS. A group of three proposes one chat at a time and prices the rows it submits, as on main.
pub(super) fn joined(chats:usize)->bool {chats==2 || (4..=PAIR_CHATS).contains(&chats)}
const _:()=assert!(PAIR_CHATS*(super::trial::TRIAL_DEPTH+1)<=VERIFY_ROWS && PAIR_CHATS<=crate::bounded_decoder::BATCH_ROWS);
/// The words every chat is expected to keep must be worth this share of the time a drafting tick adds, or more.
pub(super) const PAIR_FLOOR:f64=1.05;
/// The shortest answer that keeps its draft state beside another chat: the ticks that time the group and one look.
/// A shorter answer decodes plain rows.
pub(super) const GROUP_ROOM:usize=32;
/// Plain ticks of a group that are not timed (slow after a prompt or a pass), and the timed ones before its first look.
pub(super) const PLAIN_SKIP:u32=2;
pub(super) const PLAIN_SAMPLES:u32=2;
/// The first passes of a drafting that are not its price.
pub(super) const PASS_SKIP:u32=2;
/// THE NET: the share of the group's plain tokens so far that a chat may be behind its plain pace for looking.
/// MEASURED (run draftm-fair-f-1): a proposal that nothing checks costs every chat of two 4.5 to 7 ms of a 64 ms
/// period. With this share a group of two that never drafts looks about every 30 ticks.
pub(super) const LOOK:f64=0.004;
/// What one chat's look costs every chat, in plain periods, before the group has timed one.
pub(super) const LOOK_GUESS:f64=0.15;
/// THE NET: the tokens a chat may lose to a drafting, counted from its start. One tick that keeps a chat's own row
/// only costs it more (a pass of 8 rows takes 2.3 plain periods), so one such tick is the last; a tick that keeps
/// two rows of four costs half of it.
pub(super) const ROOM:f64=1.0;
/// A chat's answer text is under way this many steps after its think block ended (the lone trial's ANSWER_SKIP: the
/// opening of an answer says more than the text will keep), or after THINK_MOST steps with no end of a think block
/// (a request that does not think shows none). MEASURED 2026-11-17: the think blocks of the lab's three structured
/// prompts end at tokens 89, 114 and 10, of its code prompts at 39 and 0, of its prose prompts at 0.
pub(super) const TEXT_SKIP:u32=super::trial::ANSWER_SKIP;
pub(super) const THINK_MOST:u32=2*super::trial::THINK_WAIT;
/// A chat's own clock for looks: its proposals asked so far, and the one at which its think block ended; its request's
/// cap of new tokens and the history it had at its first proposal (`left`).
#[derive(Clone,Copy,Default)]
pub(super) struct Text {steps:u32,began:Option<u32>,cap:Option<usize>,base:Option<usize>}
impl Text {
    /// The clock of a request of at most `cap` new tokens.
    pub(super) fn new(cap:usize)->Self {Self {cap:Some(cap),..Self::default()}}
    /// The tokens the request has left before its cap, when its history has `history` tokens: its first proposal's
    /// history is its prompt and the tokens it had then; the token chosen and not yet in the history is counted too.
    /// (A chat whose first proposal came later counts the tokens before it as not made: its count is high, never low.)
    pub(super) fn left(&mut self,history:usize)->Option<usize> {let base=*self.base.get_or_insert(history);self.cap.map(|cap|cap.saturating_sub(history.saturating_sub(base)+1))}
    /// The chat's next row takes `next`. Whether its answer text is under way. `marker`: the token that ends a think
    /// block; a service that names none has answer text from the first step.
    pub(super) fn reads(&mut self,next:u32,marker:Option<u32>)->bool {
        self.steps+=1;
        // The answer began at the marker. With no marker in THINK_MOST steps it is taken as under way from that step
        // on; a marker that came before then is waited for in full (TEXT_SKIP), also when THINK_MOST comes first.
        if self.began.is_none() {
            if marker.is_none_or(|marker|marker==next) {self.began=Some(self.steps);}
            else if self.steps>=THINK_MOST {self.began=Some(self.steps.saturating_sub(TEXT_SKIP));}
        }
        self.began.is_some_and(|began|self.steps>=began+TEXT_SKIP)
    }
}
/// THE NET: a chat that is ahead of its plain pace banks at most this many tokens.
pub(super) const NET:f64=3.0;
/// THE NET counts no tick as longer than this many plain periods: a stall of the whole service is not a chat's loss.
/// A tick with a pass has a wider limit by its rows (`Pair::stall`): a real tick must stay under the limit, or the net
/// sees less loss than the clock. MEASURED: a tick of three chats that check their whole depth takes 3.1 to 3.5 plain
/// periods, and the first pass of such a drafting 3.94.
pub(super) const STALL:f64=3.0;
/// THE NET counts the plain ticks right after a drafting at their real time: they are slower than the plain period
/// (MEASURED, run draftm-text-j-1: 98.9, 75.7, 69.4, 66.1 ms after a drafting of the code pair, plain 63.6), and that
/// is the drafting's cost. They are no samples of the plain period or of a look's cost.
pub(super) const COOL:u32=4;
/// No look and no drafting start while a chat of the group has fewer tokens than this left before its cap: a drafting
/// must have room to pay back what its start and its end cost every chat (MEASURED 2026-11-17, run draftm-text-j-1:
/// 1.5 tokens a chat). MEASURED (runs draftm-order-o-1, draftm-27try-t-1, draftm-fair-t-1, draftm-fair-e-1): the code
/// pair began its drafting with 25 to 56 tokens left in 10 of 12 loads of 256 tokens, and a chat lost 1.1 to 2.1
/// percent in 8 of them; 2 structured began with 125 to 215 left and 3 structured with 110 to 126.
pub(super) const LATE:usize=64;
/// THE NET's limit on a pass tick (`Pair::stall`) is a period for each row a chat has in the pass and this much outside
/// it. MEASURED: real drafting ticks of four rows a chat take 2.5 (2 chats), 3.1 to 3.94 (3 chats) and 3.75 (4 chats,
/// draftm-policy29-1) plain periods, all under 4.5. With a whole period outside, one stall of the box (RB1's run
/// rb1-g16-b1, s3d-D: a first pass of 12 rows at 365.2 ms, plain 72.6 ms, every word kept) cost a chat 1.0 token, all
/// of ROOM: the drafting ended and the group never looked again in that load (its allowance could not pay back a token
/// within 256 tokens). Now one stall costs at most half a token a chat, and losses that add up to ROOM still end it.
pub(super) const PASS_OUTSIDE:f64=0.5;
/// A pass this much above its row count's price is held back until the next pass of that row count: a stall is not a
/// price. MEASURED (run draftm-27try-t-1, three chats): passes of 12 rows of 203 to 215 ms, then one of 308.7 ms;
/// its share in the price ended the drafting, and seven looks then failed by that price.
pub(super) const SLOW:f64=1.25;
/// What a group counted, for the served log, the check tools and the tests.
#[derive(Clone,Copy,Debug,Default,PartialEq)]
pub struct PairAccount {
    /// Ticks that were a pass with a drafted row, and batch steps of all the chats.
    pub turn_ticks:u32,pub plain_steps:u32,
    /// Each chat's proposals, the proposed rows its passes checked, and the rows its cycles kept (its own row included).
    pub proposals:[u32;PAIR_CHATS],pub drafted_rows:[u64;PAIR_CHATS],pub kept_rows:[u64;PAIR_CHATS],
    /// Cycles that reached a verdict, and the sum of their ms.
    pub cycles:u32,pub turn_ms:f64,
    /// The group's timed batch step, its plain period and its time outside the pass of a drafting tick, ms. None: never timed.
    pub step_ms:Option<f64>,pub plain_ms:Option<f64>,pub outside_ms:Option<f64>,
    /// Ticks with a look, the times the group began to draft, and its drafting ticks that admitted no row.
    pub looks:u32,pub starts:u32,pub dry_ticks:u32,
    /// What one look cost every chat, in plain periods (None: never timed), and the tokens each chat was behind its plain pace.
    pub look_cost:Option<f64>,pub behind:[f64;PAIR_CHATS]}
/// The tick that runs: whether it was a step or a pass of all the chats, its pass (ms) and that pass's rows, and the
/// tokens each chat made.
#[derive(Clone,Copy,Default)]
struct Tick {whole:bool,pass:Option<f64>,rows:usize,tokens:[f64;PAIR_CHATS]}
/// The proposals for the tick that comes: the chats that were told to propose (and whether it is a chat's first
/// look), the checked rows the chats bring so far, each chat's choice (made, words, expected kept words), whether
/// the drafting ends with that tick, and the chats whose own words did not pass in it.
#[derive(Clone,Copy,Default)]
struct Next {asks:u32,first:bool,offered:usize,done:[bool;PAIR_CHATS],words:[usize;PAIR_CHATS],gain:[f64;PAIR_CHATS],ends:bool,failed:[bool;PAIR_CHATS]}
/// The chats that can draft beside another chat, and the record of those of them that are a group now.
pub(super) struct Pair {
    /// Chats that keep their draft state beside others, in the order they came.
    chats:Vec<u64>,
    /// Pending verify rows that a member will compute again in its next shared pass.
    carried:BTreeMap<u64,usize>,
    /// The timed batch step of the chats together (engine ms), their plain period (wall ms) and its samples, and the
    /// plain ticks of all the chats in a row before the tick that runs.
    step:Option<f64>,plain:Option<f64>,samples:u32,run:u32,
    /// The timed shared passes of the chats by their rows in all, ms; the wall time of a drafting tick outside its pass.
    pass:[Option<f64>;VERIFY_ROWS+1],outside:Option<f64>,
    /// When the tick that runs began (ms on the caller's clock), the tick, and the proposals for the next one.
    began:Option<f64>,tick:Tick,next:Next,
    /// The group drafts; the chats whose last look passed; the next look follows a pass or the end of a drafting
    /// and asks no allowance.
    hot:bool,passed:[bool;PAIR_CHATS],free:bool,
    /// Each chat's right looks in a row; the pace time (`paced`) when it began its answer text, last passed a look
    /// or drafted; and the pace time before which it does not look again: later than `since` after a look that failed.
    right:[u32;PAIR_CHATS],since:[f64;PAIR_CHATS],again:[f64;PAIR_CHATS],
    /// The chats that must be right twice in a row (a look of theirs failed, or their own words ended a drafting, since
    /// they last passed), and the tokens each chat has left before its cap (None: not known; `LATE`).
    doubt:[bool;PAIR_CHATS],left:[Option<usize>;PAIR_CHATS],
    /// The passes of the drafting so far: the first PASS_SKIP are not its price. The plain ticks after a drafting
    /// that THE NET still counts at their real time (`COOL`).
    warm:u32,cool:u32,
    /// A row count's first timed pass: it becomes a price only with a second one (`passed`).
    once:[Option<f64>;VERIFY_ROWS+1],
    /// The chats that have made their first look and those whose answer text is under way; the chat whose look
    /// waits for its tokens, and the words it proposed that have not come yet.
    seen:[bool;PAIR_CHATS],reads:[bool;PAIR_CHATS],checking:Option<usize>,shadow:Vec<u32>,
    /// The chance each chat's first proposed word had lately (a look sets it; in drafting it falls at once and rises
    /// by halves): what a chat that has not chosen yet in a tick is expected to bring.
    sure:[Option<f64>;PAIR_CHATS],
    /// THE NET: the plain tokens the group's time so far is worth, what one look costs every chat (plain periods),
    /// the tokens each chat is behind its plain pace, and what that was when the drafting began.
    paced:f64,look:Option<f64>,behind:[f64;PAIR_CHATS],floor:[f64;PAIR_CHATS],
    /// Test control: every chat proposes and checks its whole depth in every tick.
    always:bool,
    account:PairAccount}
impl Default for Pair {
    fn default()->Self {
        Self {chats:Vec::new(),carried:BTreeMap::new(),step:None,plain:None,samples:0,run:0,pass:[None;VERIFY_ROWS+1],outside:None,
            began:None,tick:Tick::default(),next:Next::default(),hot:false,passed:[false;PAIR_CHATS],free:false,right:[0;PAIR_CHATS],
            since:[0.0;PAIR_CHATS],again:[0.0;PAIR_CHATS],doubt:[false;PAIR_CHATS],left:[None;PAIR_CHATS],warm:0,cool:0,
            once:[None;VERIFY_ROWS+1],seen:[false;PAIR_CHATS],reads:[false;PAIR_CHATS],checking:None,shadow:Vec::new(),
            sure:[None;PAIR_CHATS],paced:0.0,look:None,behind:[0.0;PAIR_CHATS],floor:[0.0;PAIR_CHATS],always:false,account:PairAccount::default()}
    }
}
impl Pair {
    /// A chat that can draft beside another opened its request.
    pub(super) fn join(&mut self,id:u64) {if !self.chats.contains(&id) {self.chats.push(id);self.fresh();}}
    /// It ended at `now` (ms on the clock of `step_begins`). The account of the group it was in, if that group ran a
    /// tick. The last tick is closed first, so the account counts its time too.
    pub(super) fn leave(&mut self,id:u64,now:f64)->Option<(Vec<u64>,PairAccount)> {
        let at=self.chats.iter().position(|&chat|chat==id)?;
        if self.group()>0 && let Some(began)=self.began.take() {self.close((now-began).max(0.0));}
        let ended=(self.group()>0 && self.account!=PairAccount::default()).then(|| (self.chats.clone(),
            PairAccount {step_ms:self.step,plain_ms:self.plain,outside_ms:self.outside,look_cost:self.look,behind:self.behind,..self.account}));
        self.chats.remove(at);self.carried.remove(&id);self.fresh();
        ended
    }
    /// A new set of chats starts a new record: nothing of another group prices this one.
    fn fresh(&mut self) {
        *self=Self {chats:std::mem::take(&mut self.chats),carried:std::mem::take(&mut self.carried),
            always:self.always,..Self::default()};
    }
    /// Record the rows currently carried by this chat; zero clears a consumed prefix.
    pub(super) fn set_carried(&mut self,id:u64,rows:usize) {
        if rows==0 {self.carried.remove(&id);}
        else if self.chats.contains(&id) {self.carried.insert(id,rows);}
    }
    pub(super) fn set_always(&mut self,always:bool) {self.always=always;}
    /// Whether an answer of `output_cap` tokens keeps its draft state beside another chat (GROUP_ROOM).
    pub(super) fn has_room(&self,output_cap:usize)->bool {self.always || output_cap>=GROUP_ROOM}
    /// The chats that draft together: 2 to PAIR_CHATS chats. 0: one chat, or more than a group holds.
    pub(super) fn group(&self)->usize {if (2..=PAIR_CHATS).contains(&self.chats.len()) {self.chats.len()} else {0}}
    /// Whether three or four chats may draft in this tick: the scheduler then gives a tick the rows of a whole pass.
    pub(super) fn wide(&self)->bool {self.group()>2}
    /// Chat `id`'s place in the group, when the group is all the chats that are live (`live` sessions are open).
    fn place(&self,id:u64,live:usize)->Option<usize> {if self.group()>0 && live==self.chats.len() {self.chats.iter().position(|&chat|chat==id)} else {None}}
    /// Whether chat `id` is a chat of a group: its rows ride the group's passes and tracked steps.
    pub(super) fn member(&self,id:u64,live:usize)->bool {self.place(id,live).is_some()}
    /// Whether chat `id`'s chains end by their probability and its rows are chosen by price (`choose`).
    pub(super) fn chooses(&self,id:u64,live:usize)->bool {!self.always && self.member(id,live)}
    /// THE NET: the most tokens a chat is behind `base`. The tick that runs is not closed yet: a pass counts with its
    /// own time, what the group's ticks take outside it, and the tokens it kept.
    fn worst(&self,base:&[f64;PAIR_CHATS])->f64 {
        let open=match (self.tick.pass,self.plain) {(Some(pass),Some(plain)) if self.tick.whole=>Some(((pass+self.outside.unwrap_or(0.0))/plain).min(self.stall())),_=>None};
        (0..self.chats.len()).map(|chat|self.behind[chat]-base[chat]+open.map_or(0.0,|pace|pace-self.tick.tokens[chat])).fold(f64::MIN,f64::max)
    }
    /// THE NET's limit on the tick that runs, in plain periods: STALL; for a pass, a period for each row a chat has in
    /// it and PASS_OUTSIDE outside it, when that is more. (A pass of four rows a chat takes under four steps of the
    /// chats, or drafting would never pay; what is longer is a stall.)
    fn stall(&self)->f64 {if self.tick.pass.is_some() {STALL.max(self.tick.rows as f64/self.chats.len().max(1) as f64+PASS_OUTSIDE)} else {STALL}}
    /// A chat of the group has fewer than LATE tokens left before its cap.
    fn late(&self)->bool {self.left[..self.chats.len()].iter().any(|left|left.is_some_and(|left|left<LATE))}
    /// Chat `id` has `left` tokens left before its cap (None: not known).
    pub(super) fn left(&mut self,id:u64,live:usize,left:Option<usize>) {if let Some(at)=self.place(id,live) {self.left[at]=left;}}
    /// THE NET: the tokens the chat that is furthest behind may still lose to looking.
    fn spare(&self)->f64 {LOOK*self.paced-self.worst(&[0.0;PAIR_CHATS])}
    /// Chat `id`, whose whole depth is `whole` and whose next row takes the token `next`, is about to propose beside
    /// another chat: the words it proposes in this tick. `reads`: its answer text is under way (`Text`). 0 (no
    /// proposal): before the group's plain period is timed, while a look waits for its tokens, while the look is
    /// another chat's or a chat waits after a look that failed by its price, and while THE NET leaves no room.
    pub(super) fn ask(&mut self,id:u64,live:usize,whole:usize,next:u32,reads:bool)->usize {
        let Some(at)=self.place(id,live) else {return 0;};
        self.follow(at,next);if reads && !self.reads[at] {self.since[at]=self.paced;}self.reads[at]=reads;
        if whole==0 {return 0;}
        if !self.always {
            if self.samples<PLAIN_SAMPLES || self.step.is_none() {return 0;}
            let n=self.chats.len();
            // A chat lost ROOM tokens to this drafting: it ends, and the next look waits for its allowance.
            if self.hot && self.worst(&self.floor)>ROOM {self.hot=false;self.cool=COOL;self.passed=[false;PAIR_CHATS];self.right=[0;PAIR_CHATS];self.free=false;}
            if !self.hot {
                // One look a tick, and none while a look waits for its tokens.
                if self.checking.is_some() || self.next.asks>0 {return 0;}
                // No look when a drafting could not start (`LATE`).
                if self.late() {return 0;}
                // The look is a chat's that has not passed and whose answer text is under way: the one whose look
                // failed last, when its wait is over (no chat looks before); else the first. It is paid from the
                // allowance, unless it follows a look that was right or the end of a drafting.
                let pick=(0..n).filter(|&chat|!self.passed[chat] && self.reads[chat]).reduce(|pick,chat|if self.again[chat]>self.again[pick] {chat} else {pick});
                if pick!=Some(at) || self.paced<self.again[at] {return 0;}
                if !(self.free || self.spare()>=self.look.unwrap_or(LOOK_GUESS)) {return 0;}
                self.free=false;self.next.first= !self.seen[at];self.seen[at]=true;
            }
        }
        self.next.asks+=1;whole
    }
    /// Chat `at` takes the token `next`: its look that waits for its tokens is right while they are the words it
    /// proposed. The last of them makes the look right; another token fails it, and the chat waits (`again`). A right
    /// look passes the chat, the second one in a row when a look of the chat failed since it last passed.
    fn follow(&mut self,at:usize,next:u32) {
        if self.checking!=Some(at) {return;}
        let right=self.shadow.first()==Some(&next);
        if right {self.shadow.remove(0);if !self.shadow.is_empty() {return;}}
        if right {self.right[at]+=1;} else {self.right[at]=0;self.failed_look(at);}
        self.passed[at]=self.right[at]>u32::from(self.doubt[at]);self.free=right;if self.passed[at] {self.since[at]=self.paced;self.doubt[at]=false;}
        eprintln!("SERVED PAIR LOOK session={} price=1 right={} left_words={} passed={:?}",self.chats[at],u8::from(right),self.shadow.len(),&self.passed[..self.chats.len()]);
        self.checking=None;self.shadow.clear();
    }
    /// Chat `at`'s look failed, by its price or by a wrong word: it must be right twice in a row, and it waits as long
    /// again as it has been in its answer text without a look that passed, TEXT_SKIP periods at the least (a chat
    /// whose first look fails in the tick its answer text begins has waited no time yet).
    fn failed_look(&mut self,at:usize) {self.doubt[at]=true;self.again[at]=(2.0*self.paced-self.since[at]).max(self.paced+f64::from(TEXT_SKIP));}
    /// The words of chat `id`'s look, which its price passed: the look waits for the tokens the chat decodes next.
    pub(super) fn shadow(&mut self,id:u64,live:usize,words:&[u32]) {
        if self.place(id,live).is_some_and(|at|self.checking==Some(at)) && self.shadow.is_empty() {self.shadow.extend_from_slice(words);}
    }
    /// What a pass of `rows` rows of all the chats costs, ms. One row a chat is the timed step. Above it the timed
    /// shared passes are joined by straight lines, and past the largest one the line from the step goes on. Before
    /// any pass is timed an added row is priced at half a chat's row of the step: a guess that the first timed pass
    /// replaces, low enough that sure chats bring the rows that time it.
    fn rows_ms(&self,rows:usize)->Option<f64> {
        let (chats,step)=(self.chats.len(),self.step?);
        if rows<=chats {return Some(step);}
        let rows=rows.min(VERIFY_ROWS);let (mut low,mut high)=((chats,step),None);
        for count in chats+1..=VERIFY_ROWS {if let Some(ms)=self.pass[count] {if count<=rows {low=(count,ms);} else if high.is_none() {high=Some((count,ms));}}}
        if low.0==rows {return Some(low.1);}
        let slope=match high {Some((count,ms))=>(ms-low.1)/(count-low.0) as f64,None if low.0>chats=>((low.1-step)/(low.0-chats) as f64).max(0.0),None=>step/(2*chats) as f64};
        Some(low.1+slope*(rows-low.0) as f64)
    }
    /// Chat `id` made a proposal in `proposal_ms`. `rates[k]`: the calibrated chance that its word `k+1` is kept when
    /// the words before it are. Returns the words its rows of this tick's pass check. A LOOK checks none: its price
    /// passes when the chat's whole proposal would pay beside chats as sure as it is. While the group DRAFTS: the count
    /// with the most expected kept words beyond the time it adds, among the counts that every chat of the group
    /// gains by (the top of this file).
    pub(super) fn choose(&mut self,id:u64,live:usize,rates:&[f64],proposal_ms:f64)->usize {
        let most=rates.len();
        let (Some(at),Some(step),Some(plain))=(self.place(id,live),self.step,self.plain) else {return self.offer(id,live,most);};
        let (n,look)=(self.chats.len(),!self.hot);
        // Carried rows price a joined group (two or four chats); a group of three prices its submitted rows, as on main.
        let carried=if joined(n) {self.carried.values().sum::<usize>()} else {0};
        // gains[w]: the words this chat is expected to keep of its first `w`.
        let mut gains=vec![0.0;most+1];let mut chain=1.0;
        for (word,rate) in rates.iter().enumerate() {chain*=rate.clamp(0.0,1.0);gains[word+1]=gains[word]+chain;}
        // Before a drafting tick is timed: the plain tick's own time outside its step, and the proposal cost
        // (one batched command for a joined group; per-chat serial proposals otherwise).
        let outside=self.outside.unwrap_or((plain-step).max(0.0)+if joined(n) {proposal_ms} else {proposal_ms*n as f64});
        let (mut best,mut value)=(0,0.0);
        // A look asks about the whole proposal only.
        for (words,&gain) in gains.iter().enumerate().skip(if look {most.max(1)} else {1}) {
            // The other chats: what each chose in this tick; else as many words as this chat, at the chance its own
            // first words had lately. A chat that never proposed here is not sure. In a look: chats as sure as this one.
            let (mut rows,mut least)=(n+words,gain);
            for other in (0..n).filter(|&other|other!=at) {
                let (brings,keeps)=if look {(words,gain)} else if self.next.done[other] {(self.next.words[other],self.next.gain[other])}
                    else {(words,self.sure[other].map_or(0.0,|sure|(1..=words).map(|word|sure.powi(word as i32)).sum::<f64>()))};
                rows+=brings;least=least.min(keeps);}
            let added=self.rows_ms(rows+carried).unwrap_or(step)+outside-plain;
            if least*plain>=PAIR_FLOOR*added && gain-added/plain>value {best=words;value=gain-added/plain;}
        }
        // What the others expect of this chat next: what its look saw; while it drafts, a lower chance at once and a
        // higher one by halves.
        if let Some(&first)=rates.first() {self.sure[at]=Some(match self.sure[at] {Some(old) if !look && first>old=>0.5*old+0.5*first,_=>first});}
        self.account.proposals[at]+=1;
        if look {
            // The price passed: the look now waits for its tokens (`shadow`, `follow`). Else it failed, and the chat
            // waits as long again as it has been in its answer text without a look that passed.
            if best>0 {self.checking=Some(at);self.shadow.clear();} else {self.passed[at]=false;self.free=false;self.right[at]=0;self.failed_look(at);}
            eprintln!("SERVED PAIR {} session={id} price={} words={most} expects={:.2} first={:.2}",if best>0 {"PRICE"} else {"LOOK"},u8::from(best>0),gains[most],rates.first().copied().unwrap_or(0.0));
            best=0;
        } else if best==0 {
            // The drafting ends with this tick. This chat's own words did not pass if they would not pay beside chats
            // as sure as it is; else another chat's did not. When another chat's rows are in the tick already, this
            // chat checks the words that leave it better off than riding on them, and no more words than that chat
            // brought: it priced as many rows of this chat.
            self.next.ends=true;
            self.next.failed[at]= !gains.iter().enumerate().skip(1).any(|(words,gain)|gain*plain>=PAIR_FLOOR*(self.rows_ms(n*(words+1)+carried).unwrap_or(step)+outside-plain));
            let (base,most)=(n+self.next.offered,most.min(self.next.words[..n].iter().copied().max().unwrap_or(0)));
            let ride=self.rows_ms(base+carried).unwrap_or(step);
            for (words,gain) in gains.iter().enumerate().take(most+1).skip(1) {let worth=gain-(self.rows_ms(base+words+carried).unwrap_or(ride)-ride)/plain;if worth>value {best=words;value=worth;}}
        }
        self.next.done[at]=true;self.next.words[at]=best;self.next.gain[at]=gains[best];self.next.offered+=best;
        best
    }
    /// Chat `id` made a proposal of `words` words with no price asked (the test control, a missing probability, DRAFT
    /// CONFIDENCE off). Returns the words its rows of this tick's pass check. A LOOK checks none: like every look, it
    /// waits for the tokens the chat decodes next (`shadow`, `follow`), and only a look whose words come passes. While
    /// the group drafts, and with the test control, all of them, and THE NET alone guards.
    pub(super) fn offer(&mut self,id:u64,live:usize,words:usize)->usize {
        let Some(at)=self.place(id,live) else {return words;};
        self.account.proposals[at]+=1;
        if !self.hot && !self.always {self.checking=Some(at);self.shadow.clear();return 0;}
        self.next.done[at]=true;self.next.words[at]=words;self.next.gain[at]=words as f64;self.next.offered+=words;
        words
    }
    /// Keep plain-step predictor inputs for the group: a batch step of all its chats then leaves each proposer what
    /// it needs for its next proposal. MEASURED (run draftm-fair-f-1): such a step costs what a plain one costs.
    pub(super) fn tracking(&self,ids:&[u64])->bool {
        !self.always && self.group()>0 && ids.len()==self.chats.len() && ids.iter().all(|id|self.chats.contains(id))
    }
    /// The scheduler's step of this tick starts at `now` (ms on one clock). `wide`: a chat of the group brings more
    /// than its own row. The tick before it is closed: its wall time prices the group and counts each chat's tokens
    /// against its plain pace. Whether the step is a pass of the group (shared.rs); else its chats ride the batch step.
    pub(super) fn step_begins(&mut self,wide:bool,now:f64)->bool {
        if let Some(began)=self.began.replace(now) {self.close((now-began).max(0.0));}
        if self.next.asks>0 && !self.always {
            // A drafting ends with a tick in which a chat's words did not pass. A chat whose own words failed must
            // pass a look again, the first at once; when no chat's own words failed, every chat must.
            if self.hot {if self.next.ends || self.next.offered==0 {
                // A chat whose own words failed is doubtful and no look is free: it looks when the allowance has room. The
                // other chats keep their pass, as of now.
                self.hot=false;self.cool=COOL;self.free=false;let own=self.next.failed.contains(&true);
                for chat in 0..PAIR_CHATS {
                    self.passed[chat]&=own && !self.next.failed[chat];if !self.passed[chat] {self.right[chat]=0;}
                    if self.next.failed[chat] {self.doubt[chat]=true;}
                }
                self.since=[self.paced;PAIR_CHATS];
                if self.next.offered==0 {self.account.dry_ticks+=1;}}}
            else {self.account.looks+=1;}
        }
        // A drafting starts when every chat's last look has passed, while it has room (`LATE`). It is judged from here (`floor`).
        if !self.hot && !self.always && self.group()>0 && self.passed[..self.chats.len()].iter().all(|&passed|passed) && !self.late() {self.hot=true;self.free=false;self.floor=self.behind;self.warm=0;self.account.starts+=1;}
        let turns=self.group()>0 && if self.always {self.next.asks>0} else {wide};
        self.next=Next::default();self.tick=Tick {tokens:[1.0;PAIR_CHATS],..Tick::default()};if turns {self.account.turn_ticks+=1;}
        turns
    }
    /// The tick that ran took `period` ms on the wall clock, the proposals for the next tick included.
    fn close(&mut self,period:f64) {
        if !self.tick.whole {self.run=0;return;}
        let asks=self.next.asks;
        // A plain tick right after a drafting (`COOL`).
        let cooling=self.tick.pass.is_none() && self.cool>0;if cooling {self.cool-=1;}
        match (self.tick.pass,self.plain) {
            // A plain tick after PLAIN_SKIP of them in a row: with no proposal inside it, it is the plain period;
            // with proposals that nothing checked, what it took longer is what one of them costs every chat. (A
            // chat's first proposal starts its draft head and costs more: it is not the price of the next look.)
            (None,plain)=>{self.run+=1;
                if self.run>PLAIN_SKIP && !cooling {match plain {
                    _ if asks==0=>{self.samples+=1;self.plain=Some(plain.map_or(period,|old|old+(period-old)/f64::from(self.samples.min(4))));},
                    Some(plain) if !self.next.first=>{let cost=((period-plain)/plain).max(0.0)/f64::from(asks);self.look=Some(self.look.map_or(cost,|old|0.5*old+0.5*cost));},
                    _=>{}}}},
            (Some(pass),_)=>{self.run=0;let outside=(period-pass).max(0.0);if self.warm>PASS_SKIP {self.outside=Some(self.outside.map_or(outside,|old|0.75*old+0.25*outside));}}}
        // THE NET: the time so far in plain tokens, and each chat's tokens of a tick that proposed or checked, or that
        // follows a drafting (`COOL`), against its plain pace. A stall of the service is not a chat's loss. Any other
        // plain tick with no proposal is on pace.
        let Some(plain)=self.plain else {self.paced+=1.0;return;};
        let touched=asks>0 || self.tick.pass.is_some() || cooling;let pace=if touched {period.min(self.stall()*plain)/plain} else {1.0};
        self.paced+=pace;
        if touched && !self.always {for (behind,tokens) in self.behind[..self.chats.len()].iter_mut().zip(&self.tick.tokens) {*behind=(*behind+pace-tokens).max(-NET);}}
    }
    /// Chat `id`'s cycle of this tick checked `drafted` proposed rows and kept `kept` rows, its own row included, in
    /// `ms` (its proposal, its part of the pass by its rows, its verdict).
    pub(super) fn turned(&mut self,id:u64,live:usize,drafted:u64,kept:u64,ms:f64) {
        if drafted==0 {return;}let Some(at)=self.place(id,live) else {return;};
        let account=&mut self.account;account.cycles+=1;account.turn_ms+=ms;account.drafted_rows[at]+=drafted;account.kept_rows[at]+=kept;
        self.tick.tokens[at]=kept.max(1) as f64;
    }
    /// The chats of the group rode one step of all of them in `ms`.
    pub(super) fn stepped(&mut self,ids:&[u64],ms:f64) {
        if self.group()==0 || ids.len()!=self.chats.len() || !ids.iter().all(|id|self.chats.contains(id)) {return;}
        self.account.plain_steps+=1;self.tick.whole=true;self.tick.pass=None;
        // The first steps after a prompt or a pass are slower: only a step after PLAIN_SKIP steps in a row is timed.
        if self.run>=PLAIN_SKIP {self.step=Some(self.step.map_or(ms,|old|0.75*old+0.25*ms));}
    }
    /// One shared pass of `chats` chats computed `rows` rows, including carry, in `ms`.
    /// Return the width charged to the group, or zero if this pass is outside its account.
    pub(super) fn passed(&mut self,chats:usize,rows:usize,ms:f64)->usize {
        if chats!=self.chats.len() || rows<=chats || rows>VERIFY_ROWS {return 0;}
        self.tick.whole=true;self.tick.pass=Some(ms);self.tick.rows=rows;self.warm+=1;
        // A row count timed once is no price yet: one slow pass must not refuse that row count for the group's life
        // (a refused row count is never timed again). Its second timed pass makes the price, the mean of the two.
        // A pass far above the price (`SLOW`) is held back the same way: the next pass of that row count confirms it,
        // or it was a stall and is dropped. The first two timed passes make the price when they agree within SLOW;
        // else the lower one does (MEASURED, run draftm-nine-b-1: 206.1 and 285.4 ms made a first price of 245.8, at
        // which three chats that kept every word did not pay). A row count that is really slower shows it again.
        if self.warm>PASS_SKIP {self.pass[rows]=match (self.pass[rows],self.once[rows].take()) {
            (Some(old),Some(first)) if ms>SLOW*old=>Some(0.5*old+0.25*first+0.25*ms),
            (Some(old),None) if ms>SLOW*old=>{self.once[rows]=Some(ms);Some(old)},
            (Some(old),_)=>Some(0.75*old+0.25*ms),
            (None,Some(first)) if first.max(ms)>SLOW*first.min(ms)=>Some(first.min(ms)),
            (None,Some(first))=>Some(0.5*first+0.5*ms),(None,None)=>{self.once[rows]=Some(ms);None}};}
        self.tick.rows
    }
}
/// A chat's Pay and Deeper records (price.rs, deeper.rs) while a cycle of a shared pass is open (CLAIM A-5). The
/// price of a pass the chat shared, and its peers' rows in it, are not its own: that cycle's Pay and Deeper feedback
/// goes to the group's record only (`Pair::turned`), and the two records are put back as they were when its rows ran
/// in the pass, once that cycle has its verdict or none; a chat that rode the pass with its own row only gets them
/// back at once. A request that never was in a shared pass never holds them: their contents move as before, and it
/// checks an empty holder twice a proposal. Other feedback is not held:
/// the gate's begun count, its stop score and its account count shared cycles, the factory's kept-share tables learn
/// from them, and a look counts as an unused proposal in Pay.
#[derive(Default)]
pub(super) struct Held(Option<(super::price::Pay,super::deeper::Deeper)>);
impl Held {
    /// The chat is in a shared pass: its records as they are now (the first hold is kept until `back`).
    pub(super) fn hold(&mut self,gate:&super::gate::Gate) {self.0.get_or_insert((gate.pay,gate.deeper));}
    /// That cycle ended: its records are as they were before it.
    pub(super) fn back(&mut self,gate:&mut super::gate::Gate) {if let Some((pay,deeper))=self.0.take() {gate.pay=pay;gate.deeper=deeper;}}
}
/// The clock of the group's ticks: ms since the service's first tick.
pub(super) fn clock_ms()->f64 {
    static ORIGIN:std::sync::OnceLock<std::time::Instant>=std::sync::OnceLock::new();
    ORIGIN.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()*1000.0
}
impl<C:super::Compute,E:super::Exchange> super::Factory<'_,C,E> {
    /// A chat leaves the group (its request ended). A group that ran prints its account.
    #[cold] #[inline(never)]
    pub(super) fn pair_leave(&self,id:u64) {
        let Some((ids,account))=self.pair.borrow_mut().leave(id,clock_ms()) else {return;};
        let mut counts=self.pair_counts.get();counts[0]+=1;counts[1]+=account.turn_ticks;counts[2]+=account.plain_steps;self.pair_counts.set(counts);
        let n=ids.len();
        eprintln!("SERVED PAIR rank={} sessions={ids:?} turn_ticks={} cycles={} plain_steps={} looks={} starts={} dry_ticks={} proposals={:?} drafted_rows={:?} kept_rows={:?} turn_ms={:.1} step_ms={:.1} plain_ms={:.1} outside_ms={:.1} look_tokens={:.3} step_timed={} behind_tokens=[{}]",
            self.split.topology.rank,account.turn_ticks,account.cycles,account.plain_steps,account.looks,account.starts,account.dry_ticks,&account.proposals[..n],&account.drafted_rows[..n],&account.kept_rows[..n],account.turn_ms,
            account.step_ms.unwrap_or(0.0),account.plain_ms.unwrap_or(0.0),account.outside_ms.unwrap_or(0.0),account.look_cost.unwrap_or(0.0),u8::from(account.plain_ms.is_some()),account.behind[..n].iter().map(|behind|format!("{behind:.1}")).collect::<Vec<_>>().join(", "));
    }
}
impl<C:super::Compute,E:super::Exchange> super::Session<'_,C,E> {
    /// GROUP DRAFTING: whether this chat is one of a group (`Pair::member`).
    pub(super) fn grouped(&self)->bool {let live=self.factory.active.get();live>1 && self.factory.pair.borrow().member(self.id,live)}
    /// GROUP DRAFTING: the words of this chat's proposal that its rows of the tick's pass check, by the draft head's
    /// probabilities `mass` and the group's rule (`Pair::choose`). The whole proposal when a probability is missing,
    /// or with DRAFT CONFIDENCE off or the pair's test control on.
    pub(super) fn group_width(&mut self,mass:&[Option<f32>],proposal_ms:f64)->usize {
        let factory=self.factory;let (live,whole)=(factory.active.get(),mass.len());
        self.checked_in_four=live==4 && factory.pair.borrow().member(self.id,live);
        if !(factory.confidence_on.get() && mass.iter().all(Option::is_some) && factory.pair.borrow().chooses(self.id,live)) {
            return factory.pair.borrow_mut().offer(self.id,live,whole);}
        // A chat reads and feeds the kept-share table of its request's kind (confidence.rs `Tables`).
        let mut tables=factory.confidence.borrow_mut();let calibration=tables.checked_for(self.alone,self.checked_in_four);
        let rates=mass.iter().map(|p|calibration.rate(p.unwrap_or(0.0))).collect::<Vec<_>>();
        let width=factory.pair.borrow_mut().choose(self.id,live,&rates,proposal_ms);
        calibration.offered+=1;calibration.checked+=width as u64;
        if width==0 {calibration.plain+=1;} else if width<whole {calibration.shortened+=1;}
        width
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test]
    fn two_chats_price_the_physical_rows_after_a_second_rollback() {
        use crate::bounded_decoder::{Host,TestCheckpoint,Verify};
        use crate::row_split::{Factory,Split,kept_charge_tests::Local,limits};
        use spark_serve::{BatchOutput,BatchRows,Engine,Predictions,Session as _};

        let checkpoint=TestCheckpoint::new_with_predictor(80);
        let factory=||Factory::new(&checkpoint.admission,Split::new(Host,Local).unwrap(),80,vec![0])
            .unwrap().with_predictor(3).unwrap();
        let group=factory();
        let solo=factory();
        let mut limits=limits(80,64).unwrap();
        limits.next_mtp_depth=Some(3);
        let request=spark_serve::protocol::parse(
            br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":64,"draft":"strict","mtp_depth":3}"#,
            true,&limits).unwrap();
        let mut chats=[group.session().unwrap(),group.session().unwrap()];
        let mut reference=[solo.session().unwrap(),solo.session().unwrap()];
        for (at,prompt) in [[3,5,1],[2,6,4]].iter().enumerate() {
            for chat in [&mut chats[at],&mut reference[at]] {
                chat.configure_request(&request).unwrap();
                chat.prefill(prompt).unwrap();
            }
        }
        // Warm the real price table. Each round first sends eight new rows, then seven new rows plus two carried.
        let mut charges=Vec::new();
        for _ in 0..3 {
            for carry in [false,true] {
                let left:&[u32]=if carry {&[1,5,2]} else {&[7,3,2,1]};
                let words=[left,&[4,1,6,2]];
                if carry {assert_eq!(chats[0].open.as_ref().unwrap().carried(),2);}
                let physical=chats.iter().zip(words).map(|(chat,row)|chat.open.as_ref()
                    .map_or(row.len(),|open|open.shared_count(row.len()))).sum::<usize>();
                assert_eq!(physical,if carry {9} else {8});
                assert_eq!(group.pair.borrow().carried.values().sum::<usize>(),if carry {2} else {0});
                let mut batch=chats.iter_mut().zip(words).map(|(session,tokens)|BatchRows {
                    session,tokens,output:BatchOutput::All,greedy:false,top_k:0,
                }).collect::<Vec<_>>();
                let replies=group.step_batch(&mut batch).map_err(|e|e.message).unwrap();
                assert!(group.pair.borrow().carried.is_empty());
                for ((reply,own),tokens) in replies.into_iter().zip(&mut reference).zip(words) {
                    let Predictions::Logits(actual)=reply else {panic!("shared verification must return logits");};
                    let expected=own.rows(tokens).unwrap();
                    assert_eq!(actual.len(),expected.len());
                    for (got,want) in actual.iter().zip(&expected) {
                        let bits=|row:&[f32]|row.iter().map(|word|word.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(got),bits(want));
                    }
                }
                charges.push((physical,group.pair.borrow().tick.rows));
                if !carry {
                    for chat in [&mut chats[0],&mut reference[0]] {
                        let end=chat.open.as_ref().map(Verify::position).unwrap() as usize;
                        chat.keep(end-1).unwrap();
                        chat.keep(end-2).unwrap();
                    }
                }
            }
        }
        let pair=group.pair.borrow();
        eprintln!("physical and charged rows {charges:?}; price learned at 7/8/9: {:?}",
            [pair.pass[7].is_some(),pair.pass[8].is_some(),pair.pass[9].is_some()]);
        // Two chats price the rows they compute: the carried rows are charged.
        assert_eq!(charges,[(8,8),(9,9)].repeat(3));
        assert!(pair.pass[8].is_some() && pair.pass[9].is_some() && pair.pass[7].is_none());
    }
    #[test]
    fn groups_of_two_and_four_choose_at_their_physical_width() {
        // Three chats price their submitted rows, as on main, so carried rows do not change their choice; two and four
        // price the rows they compute, so two carried rows make the tick too dear.
        for n in [2,3,4] {
            let chats=(1..=n as u64).collect::<Vec<_>>();
            let mut pair=Pair {
                chats:chats.clone(),step:Some(60.0),plain:Some(60.0),outside:Some(0.0),
                hot:true,sure:[Some(1.0);PAIR_CHATS],..Pair::default()
            };
            pair.pass[2*n]=Some(90.0);
            pair.pass[2*n+2]=Some(180.0);
            assert_eq!(pair.choose(1,n,&[1.0],0.0),1);
            pair.next=Next::default();
            pair.set_carried(2,2);
            assert_eq!(pair.carried.values().sum::<usize>(),2);
            assert_eq!(pair.choose(1,n,&[1.0],0.0),usize::from(!joined(n)),"{n} chats with two carried rows");
        }
    }
    #[test]
    fn net_uses_the_charged_physical_width_for_its_stall_cap() {
        for (rows,cap,debt) in [(7,4.0,1.0),(9,5.0,1.5)] {
            let mut pair=Pair {chats:vec![1,2],plain:Some(60.0),..Pair::default()};
            assert!(pair.step_begins(true,0.0));
            assert_eq!(pair.passed(2,rows,250.0),rows);
            pair.turned(1,2,3,3,125.0);
            pair.turned(2,2,3,3,125.0);
            assert_eq!(pair.stall(),cap);
            // A 4.5-period tick leaves both chats 1.5 periods behind at width nine.
            pair.close(270.0);
            assert_eq!(&pair.behind[..2],&[debt,debt]);
        }
    }
    /// A test's group and its clock: the ticks below move the clock by what they say a tick takes. Every chat's
    /// token of tick `t` is `t`, and chat `c` is in its answer text when `reads[c]`.
    struct Group {pair:Pair,now:f64,chats:usize,t:u32,reads:[bool;PAIR_CHATS]}
    impl Group {
        /// A group of `chats` chats (ids 1, 2, ...) after the plain ticks that time it: each takes `plain` ms on the
        /// wall clock, `step` ms of them in the batch step.
        fn new(chats:usize,step:f64,plain:f64)->Self {
            let mut pair=Pair::default();for id in 1..=chats as u64 {pair.join(id);}
            let mut group=Self {pair,now:0.0,chats,t:0,reads:[true;PAIR_CHATS]};
            for tick in 0..PLAIN_SKIP+PLAIN_SAMPLES+1 {
                assert!(group.ids().iter().all(|&id|group.pair.ask(id,chats,3,group.t,true)==0),"no proposal in plain tick {tick}: the group is not timed yet");
                group.plain(step,plain);}
            group
        }
        fn ids(&self)->Vec<u64> {(1..=self.chats as u64).collect()}
        /// A plain tick: one batch step of all the chats, and nobody is asked.
        fn plain(&mut self,step:f64,period:f64) {assert!(!self.pair.step_begins(false,self.now));let ids=self.ids();self.pair.stepped(&ids,step);self.now+=period;self.t+=1;}
        /// Plain ticks until THE NET has room for a look. Returns how many.
        fn rest(&mut self,step:f64,period:f64)->u32 {
            let mut ticks=0;while self.pair.spare()<self.pair.look.unwrap_or(LOOK_GUESS) {self.plain(step,period);ticks+=1;assert!(ticks<2000);}ticks
        }
        /// One tick. Every chat is asked; a chat that proposes has the chances `rates[chat]` for its words (cut to
        /// the depth it was asked for) and its proposal takes 4 ms a word. The first `kept[chat]` of its words are
        /// the tokens that come, the others are not. A tick with a checked word is a pass of 11 ms a row over the
        /// step, in which each chat keeps its own row and the right ones of its checked words. Returns the words
        /// each chat proposed and checked.
        fn tick(&mut self,rates:&[&[f64]],kept:&[u64],step:f64,plain:f64)->(Vec<usize>,Vec<usize>) {
            let ids=self.ids();let mut spent=0.0f64;let (mut asked,mut words)=(Vec::new(),Vec::new());
            for (at,&id) in ids.iter().enumerate() {
                let depth=self.pair.ask(id,self.chats,3,self.t,self.reads[at]).min(rates[at].len());asked.push(depth);
                words.push(if depth==0 {0} else {spent+=4.0*depth as f64;
                    let checked=self.pair.choose(id,self.chats,&rates[at][..depth],4.0*depth as f64);
                    let proposal=(1..=depth as u32).map(|word|if u64::from(word)<=kept[at] {self.t+word} else {u32::MAX}).collect::<Vec<_>>();
                    if checked==0 {self.pair.shadow(id,self.chats,&proposal);}checked});}
            // The proposals are made before the step of their tick begins: their time is in the tick before.
            let rows=self.chats+words.iter().sum::<usize>();self.now+=spent;
            if self.pair.step_begins(rows>self.chats,self.now) {
                let pass=step+11.0*(rows-self.chats) as f64;self.pair.passed(self.chats,rows,pass);
                for (at,&id) in ids.iter().enumerate() {if words[at]>0 {self.pair.turned(id,self.chats,words[at] as u64,1+kept[at].min(words[at] as u64),4.0+pass*(1+words[at]) as f64/rows as f64+1.0);}}
                self.now+=pass+2.0+(plain-step);
            } else {self.pair.stepped(&ids,step);self.now+=plain;}
            self.t+=1;(asked,words)
        }
        /// Ticks until the group drafts: what each chat was asked for in each of them.
        fn until_hot(&mut self,rates:&[&[f64]],kept:&[u64],step:f64,plain:f64)->Vec<Vec<usize>> {
            let mut asked=Vec::new();while !self.pair.hot {let (depths,words)=self.tick(rates,kept,step,plain);assert!(words.iter().all(|&words|words==0) && asked.len()<60,"{asked:?}");asked.push(depths);}asked
        }
    }
    const SURE:&[f64]=&[0.97,0.97,0.97];const PROSE:&[f64]=&[0.5,0.5,0.5];
    #[test] fn a_group_is_two_to_four_drafting_chats_alone_on_the_server() {
        let mut one=Pair::default();one.join(7);
        assert_eq!(one.ask(7,1,3,0,true),0,"one chat is not a group");assert_eq!(one.ask(7,2,3,0,true),0,"the other live chat is not a drafting chat");
        let mut pair=Pair::default();pair.join(7);pair.join(9);
        assert!(pair.member(7,2) && pair.member(9,2) && !pair.member(8,2) && !pair.member(7,3));
        // No proposal before the group's plain period is timed: PLAIN_SKIP ticks are not timed, PLAIN_SAMPLES are.
        let (mut now,mut steps)=(0.0,0);
        for tick in 0..PLAIN_SKIP+PLAIN_SAMPLES+1 {
            assert_eq!((pair.ask(7,2,3,0,true),pair.ask(9,2,3,0,true)),(0,0),"tick {tick}");assert!(!pair.step_begins(false,now));pair.stepped(&[7,9],63.0);now+=64.0;steps+=1;
            assert_eq!(pair.step.is_some(),tick>=PLAIN_SKIP,"the step of tick {tick}");}
        assert_eq!((pair.plain,pair.samples,pair.step),(Some(64.0),PLAIN_SAMPLES,Some(63.0)));
        // Then no look before THE NET has room for one: LOOK of the plain tokens so far must cover what a look costs.
        while pair.spare()<LOOK_GUESS {assert_eq!((pair.ask(7,2,3,0,true),pair.ask(9,2,3,0,true)),(0,0));assert!(!pair.step_begins(false,now));pair.stepped(&[7,9],63.0);now+=64.0;steps+=1;}
        assert_eq!(pair.paced,(LOOK_GUESS/LOOK).ceil(),"the first look comes after 38 plain ticks");
        assert_eq!(pair.ask(8,2,3,0,true),0,"a chat that is not of the group");assert_eq!(pair.ask(7,3,3,0,true),0,"a third live chat that is not of the group: plain rows, as before");
        // The look is one chat's: the first in line whose answer text is under way proposes its whole depth. Here the
        // first chat still thinks, so the second looks.
        assert_eq!((pair.ask(7,2,3,0,false),pair.ask(9,2,0,0,true),pair.ask(9,2,3,0,true)),(0,0,3),"a chat that thinks, a chat of depth 0, the chat that looks");
        // A tick is a pass of the group only when a chat brings a checked row.
        let mut wide=Pair::default();wide.join(7);wide.join(9);assert!(!wide.step_begins(false,0.0));assert!(wide.step_begins(true,1.0));
        // A step of other chats is not the group's step.
        let before=pair.step;pair.stepped(&[7,8],90.0);pair.stepped(&[7],40.0);assert_eq!(pair.step,before);
        // A chat that leaves ends the group, and the next group starts a new record.
        assert!(pair.step_begins(true,now));pair.passed(2,8,130.0);pair.turned(7,2,3,4,75.0);
        let (ids,account)=pair.leave(9,now+152.0).unwrap();assert_eq!(ids,[7,9]);
        assert_eq!((account.turn_ticks,account.plain_steps,account.cycles,&account.kept_rows[..4],&account.drafted_rows[..4],account.looks),(1,steps,1,&[4,0,0,0][..],&[3,0,0,0][..],1));assert_eq!((account.step_ms,account.plain_ms),(Some(63.0),Some(64.0)));
        assert_eq!(pair.ask(7,2,3,0,true),0);pair.join(11);assert!(pair.step.is_none() && pair.plain.is_none() && pair.paced==0.0 && pair.seen==[false;PAIR_CHATS] && pair.pass.iter().all(Option::is_none));assert!(pair.leave(11,now+200.0).is_none(),"a group that ran no tick prints nothing");
        let mut four=Group::new(4,78.0,79.2);
        assert!(four.pair.wide() && four.pair.tracking(&[1,2,3,4]) && (1..=4).all(|id|four.pair.member(id,4)));
        assert!(!four.pair.member(1,5));
        let looks=four.until_hot(&[SURE;4],&[3;4],78.0,79.2);
        for at in 0..4 {assert_eq!(looks.iter().filter(|depths|depths[at]>0).count(),1);}
        assert_eq!(four.tick(&[SURE;4],&[3;4],78.0,79.2),(vec![3;4],vec![3;4]));
        assert_eq!(four.pair.tick.rows,16);
        let mut eight=Group::new(PAIR_CHATS,95.0,96.5);
        let eight_ids=(1..=PAIR_CHATS as u64).collect::<Vec<_>>();
        assert!(eight.pair.wide() && eight.pair.tracking(&eight_ids) && eight_ids.iter().all(|&id|eight.pair.member(id,PAIR_CHATS)));
        let looks=eight.until_hot(&[SURE;PAIR_CHATS],&[3;PAIR_CHATS],95.0,96.5);
        for at in 0..PAIR_CHATS {assert_eq!(looks.iter().filter(|depths|depths[at]>0).count(),1);}
        assert_eq!(eight.tick(&[SURE;PAIR_CHATS],&[3;PAIR_CHATS],95.0,96.5),(vec![3;PAIR_CHATS],vec![3;PAIR_CHATS]));
        assert_eq!(eight.pair.tick.rows,PAIR_CHATS*4);
        // PAIR_CHATS + 1 live drafting chats exceed the group cap.
        let over_ids=(1..=(PAIR_CHATS+1) as u64).collect::<Vec<_>>();
        let mut over=Pair::default();for &id in &over_ids {over.join(id);}
        assert_eq!(over.group(),0);assert!(!over.wide() && !over.tracking(&over_ids));
        for tick in 0..60 {assert!(!over.step_begins(true,64.0*f64::from(tick)));over.stepped(&over_ids,82.0);for &id in &over_ids {assert_eq!(over.ask(id,over_ids.len(),3,0,true),0);}}
        let three=Group::new(3,75.0,76.0).pair;assert!(three.wide() && three.tracking(&[1,2,3]) && !three.tracking(&[1,2]));
    }
    #[test] fn a_chat_is_looked_at_only_in_its_answer_text() {
        // A think block that ends at step 20 (token 9): the answer text is under way TEXT_SKIP steps later.
        let mut text=Text::default();assert_eq!((1..=60).position(|step|text.reads(if step==20 {9} else {4},Some(9))),Some((20+TEXT_SKIP) as usize-1));
        // A service that names no such token: TEXT_SKIP steps after the first. A think block with no end: THINK_MOST steps.
        let mut text=Text::default();assert_eq!((1..=60).position(|_|text.reads(4,None)),Some(TEXT_SKIP as usize));
        let mut text=Text::default();assert_eq!((1..=300).position(|_|text.reads(4,Some(9))),Some(THINK_MOST as usize-1));
        // A think block that ends late, at step THINK_MOST-8: the answer still waits its TEXT_SKIP steps, though
        // THINK_MOST comes first. And a marker after THINK_MOST does not take the answer back.
        let late=THINK_MOST-8;let mut text=Text::default();assert_eq!((1..=300).position(|step|text.reads(if step==late {9} else {4},Some(9))),Some((late+TEXT_SKIP) as usize-1));
        let mut text=Text::default();assert!((1..=300).filter(|&step|text.reads(if step==THINK_MOST+40 {9} else {4},Some(9))).eq(THINK_MOST..=300));
        // Two sure chats of which one still thinks: the other looks and passes, and then the round waits for the one
        // that thinks. No look is made of it, and the group does not draft. When it reads its answer, it looks at once.
        let mut group=Group::new(2,64.0,64.5);group.reads[0]=false;group.rest(64.0,64.5);
        let asked=(0..100).map(|_|group.tick(&[SURE,SURE],&[3,3],64.0,64.5).0).filter(|asked|asked!=&[0,0]).collect::<Vec<_>>();
        assert_eq!(asked,[[0,3]]);assert_eq!((&group.pair.passed[..2],group.pair.hot,group.pair.account.looks),(&[false,true][..],false,1));
        group.reads=[true;PAIR_CHATS];assert_eq!(group.until_hot(&[SURE,SURE],&[3,3],64.0,64.5),[[3,0],[0,0],[0,0],[0,0]]);
        // Chats that all think are not looked at.
        let mut group=Group::new(2,64.0,64.5);group.reads=[false;PAIR_CHATS];
        for _ in 0..300 {assert_eq!(group.tick(&[SURE,SURE],&[3,3],64.0,64.5).0,[0,0]);}assert_eq!((group.pair.account.looks,group.pair.behind),(0,[0.0;PAIR_CHATS]));
        // A prose chat beside a sure chat that still thinks: its look fails by its price, and it waits. When the sure
        // chat reads its answer, it does not look while the prose chat waits: the group cannot draft without that
        // chat, and a chat's first look is its dearest. The prose chat looks first, fails again, and waits longer.
        let mut group=Group::new(2,64.0,64.5);group.reads[0]=false;group.rest(64.0,64.5);
        assert_eq!(group.tick(&[SURE,PROSE],&[3,3],64.0,64.5).0,[0,3]);let wait=group.pair.again[1];assert!((76.0..80.0).contains(&wait),"{wait}");
        group.reads=[true;PAIR_CHATS];while group.pair.paced<wait {assert_eq!(group.tick(&[SURE,PROSE],&[3,3],64.0,64.5).0,[0,0]);}
        group.rest(64.0,64.5);assert_eq!(group.tick(&[SURE,PROSE],&[3,3],64.0,64.5).0,[0,3]);
        assert!(group.pair.again[1]>2.0*wait && !group.pair.seen[0] && group.pair.account.looks==2,"{:?}",group.pair.again);
    }
    #[test] fn the_plain_period_is_timed_after_the_slow_first_ticks_and_never_from_a_tick_with_a_proposal() {
        // MEASURED 2026-11-17 (draftm-cell-n-1, two chats, plain): the first steps after a prompt 85.8, 71.9, then 66.0, 66.6.
        let mut pair=Pair::default();pair.join(1);pair.join(2);let mut now=0.0;
        for (step,period) in [(85.8,86.4),(71.9,72.5),(66.0,66.6),(66.6,67.2)] {assert!(!pair.step_begins(false,now));pair.stepped(&[1,2],step);now+=period;}
        assert!(!pair.step_begins(false,now));
        assert!((pair.plain.unwrap()-66.9).abs()<1e-9 && (pair.step.unwrap()-(0.75*66.0+0.25*66.6)).abs()<1e-9,"{:?} {:?}",pair.plain,pair.step);
        // A plain tick in which a chat looked is longer by the look: it is not the plain period, and every chat is
        // behind its plain pace by it. A chat's first look starts its draft head: it is not the price of the next look.
        pair.stepped(&[1,2],64.0);now+=64.6;pair.paced=100.0;assert_eq!(pair.ask(1,2,3,0,true),3);let before=pair.plain;now+=20.0;assert!(!pair.step_begins(false,now));
        let plain=before.unwrap();assert_eq!((pair.plain,pair.look),(before,None));assert!((pair.behind[0]-(84.6/plain-1.0)).abs()<1e-9 && pair.behind[0]==pair.behind[1],"{:?}",pair.behind);
        // A later look is.
        pair.stepped(&[1,2],64.0);now+=64.6;pair.paced=1000.0;assert_eq!(pair.ask(1,2,3,0,true),3);now+=8.0;assert!(!pair.step_begins(false,now));
        assert!(pair.plain==before && (pair.look.unwrap()-(72.6-plain)/plain).abs()<1e-9,"{:?} {:?}",pair.plain,pair.look);
        // After a pass the next PLAIN_SKIP plain ticks are not timed either; the one after them is.
        let mut group=Group::new(2,64.0,64.5);group.pair.hot=true;group.pair.sure=[Some(0.97);PAIR_CHATS];
        assert_eq!(group.tick(&[SURE,SURE],&[3,3],64.0,64.5).1,[3,3]);let before=(group.pair.plain,group.pair.samples);
        for _ in 0..PLAIN_SKIP {group.plain(90.0,90.5);}assert!(!group.pair.step_begins(false,group.now));assert_eq!((group.pair.plain,group.pair.samples),before);
        group.pair.stepped(&[1,2],64.0);group.now+=70.5;assert!(!group.pair.step_begins(false,group.now));assert_eq!(group.pair.samples,before.1+1);
    }
    #[test] fn rows_are_priced_from_the_groups_own_timed_step_and_passes() {
        // MEASURED 2026-11-17 (draftm-base-m-1): the step of two chats 63.9 ms, their shared pass of 8 rows 130.6 ms.
        let mut pair=Group::new(2,63.9,64.5).pair;let near=|a:Option<f64>,b:f64|(a.unwrap()-b).abs()<1e-9;
        assert!(near(pair.plain,64.5) && near(pair.rows_ms(2),63.9) && near(pair.rows_ms(1),63.9));
        // Before any pass is timed an added row is priced at half a chat's row of the step.
        assert!(near(pair.rows_ms(4),63.9+2.0*63.9/4.0) && near(pair.rows_ms(8),63.9+6.0*63.9/4.0));
        pair.pass[8]=Some(130.6);let row=(130.6-63.9)/6.0;
        for rows in 2..=8 {assert!(near(pair.rows_ms(rows),63.9+row*(rows-2) as f64),"{rows} rows");}
        // A timed pass of fewer rows bends the line: the first drafted row of a tick costs more than the later ones.
        pair.pass[3]=Some(90.0);assert!(near(pair.rows_ms(5),90.0+2.0*(130.6-90.0)/5.0));
        // Past the largest timed pass the line from the step goes on.
        assert!(near(pair.rows_ms(10),130.6+2.0*row));
        // A pass of other chats, of one row a chat, or of too many rows is not this group's price.
        let before=pair.pass;pair.passed(3,8,1.0);pair.passed(2,2,1.0);pair.passed(2,17,1.0);assert_eq!(pair.pass,before);
    }
    #[test] fn the_group_drafts_when_every_chats_look_passes_and_a_group_that_never_does_pays_the_looks_only() {
        // Two chats that are both sure and right. The first chat's look checks nothing; the three tokens that come are
        // its words, so the second chat looks; its three tokens come too; then every tick checks the whole depth of both.
        let mut alike=Group::new(2,64.0,64.5);assert_eq!(alike.rest(64.0,64.5),34);
        assert_eq!(alike.until_hot(&[SURE,SURE],&[3,3],64.0,64.5),[[3,0],[0,0],[0,0],[0,3],[0,0],[0,0],[0,0]]);
        for tick in 0..200 {assert_eq!(alike.tick(&[SURE,SURE],&[3,3],64.0,64.5),(vec![3,3],vec![3,3]),"tick {tick}");}
        assert!(alike.pair.hot && alike.pair.behind[..2].iter().all(|&behind|behind==-NET),"both chats are ahead of their plain pace: {:?}",alike.pair.behind);
        let account=alike.pair.account;assert_eq!((account.looks,account.starts,account.dry_ticks,account.turn_ticks),(2,1,0,200));
        // A sure chat beside a prose chat, two prose chats, and a chat whose price passes while its words are wrong:
        // no row is ever admitted, so no chat ever waits for a row of the other. Only the chat whose look failed
        // looks again, and all the looks together cost each chat LOOK of its tokens.
        // A chat whose look fails, by its price or by a wrong word, waits as long again as it has gone without a
        // pass: 4 looks of one chat in 600 ticks (near tick 38, 84, 168, 336).
        for (order,kept,first,lookers,count) in [([SURE,PROSE],[3,3],vec![[3,0]],[0,3],5..=5),([PROSE,SURE],[3,3],vec![],[3,0],4..=4),
            ([PROSE,PROSE],[3,3],vec![],[3,0],4..=4),([SURE,SURE],[3,2],vec![[3,0]],[0,3],5..=5)] {
            let mut mixed=Group::new(2,64.0,64.5);let mut looks=Vec::new();
            for tick in 0..600 {let (asked,words)=mixed.tick(&order,&kept,64.0,64.5);assert_eq!(words,[0,0],"tick {tick}");if asked!=[0,0] {looks.push([asked[0],asked[1]]);}}
            assert!(looks.starts_with(&first) && looks[first.len()..].iter().all(|&asked|asked==lookers),"{looks:?}");let looks=looks.len() as u32;
            let (pair,account)=(&mixed.pair,mixed.pair.account);
            assert_eq!((account.turn_ticks,account.starts,account.looks,pair.hot),(0,0,looks,false));assert!(count.contains(&looks),"{looks} looks in 600 ticks");
            // A look costs every chat 12 ms of a 64.5 ms period, and all the looks stay inside the allowance.
            assert!((pair.look.unwrap()-12.0/64.5).abs()<0.01 && pair.behind[0]==pair.behind[1] && pair.behind[0]<=LOOK*pair.paced+0.2 && (pair.behind[0]-f64::from(looks)*12.0/64.5).abs()<0.01,"{:?} {:?} {}",pair.look,pair.behind,pair.paced);
        }
        // Two chats at a chance of 0.7 a word (code) that draft: three words each pay; at 0.6 nothing does, because
        // a tick that drafts costs its proposals and its first rows whatever it keeps. Such a tick ends the drafting.
        let primed=|chance:f64| {let mut group=Group::new(2,64.0,64.5);group.pair.hot=true;group.pair.passed=[true;PAIR_CHATS];group.pair.sure=[Some(chance);PAIR_CHATS];
            group.pair.pass[4]=Some(64.0+2.0*11.0);group.pair.pass[8]=Some(64.0+6.0*11.0);group.pair.outside=Some(22.0);group};
        let rates:&[f64]=&[0.7,0.7,0.7];let mut code=primed(0.7);assert_eq!(code.tick(&[rates,rates],&[2,2],64.0,64.5).1,[3,3]);assert!(code.pair.hot);
        let rates:&[f64]=&[0.6,0.6,0.6];let mut weak=primed(0.6);assert_eq!(weak.tick(&[rates,rates],&[2,2],64.0,64.5),(vec![3,3],vec![0,0]));assert!(!weak.pair.hot && weak.pair.account.dry_ticks==1);
        // THE FIRST PASSES OF A DRAFTING ARE NOT ITS PRICE. MEASURED 2026-11-17 (draftm-wait-k-1, three structured
        // chats): passes of 12 rows 240.9 and 201.3 ms with 57.6 ms outside them, where steady ticks take 204 and 32.
        // Priced from those two ticks, the third was refused by 0.02 token and the drafting ended.
        let mut slow=Group::new(3,75.5,75.7);slow.rest(75.5,75.7);slow.until_hot(&[SURE,SURE,SURE],&[3,3,3],75.5,75.7);
        // A ROW COUNT TIMED ONCE IS NO PRICE YET: the third pass is the first that is timed, the fourth makes the price.
        let before=slow.pair.behind[0];
        for (tick,(pass,outside)) in [(240.9,57.6),(201.3,57.6),(204.0,32.0),(204.0,32.0),(204.0,32.0)].into_iter().enumerate() {
            for id in 1..=3 {assert_eq!(slow.pair.ask(id,3,3,slow.t,true),3);assert_eq!(slow.pair.choose(id,3,SURE,10.0),3,"tick {tick}, chat {id}");}
            assert!(slow.pair.step_begins(true,slow.now));assert_eq!((slow.pair.pass[12],slow.pair.outside),(if tick>3 {Some(204.0)} else {None},if tick>2 {Some(32.0)} else {None}),"tick {tick}");
            // THE NET counts the first tick at its whole time: 298.5 ms are 3.94 plain periods for 4 tokens, not 3.
            if tick==1 {assert!((slow.pair.behind[0]-before-(298.5/75.7-4.0)).abs()<0.02,"{} {before}",slow.pair.behind[0]);}
            slow.pair.passed(3,12,pass);for id in 1..=3 {slow.pair.turned(id,3,3,4,pass/3.0);}slow.now+=pass+outside;slow.t+=1;}
        assert!(slow.pair.hot && slow.pair.account.starts==1 && slow.pair.behind[0]<0.0,"THE NET counted the slow ticks too: {:?}",slow.pair.behind);
        // ONE STALL OF THE BOX DOES NOT END A DRAFTING (PASS_OUTSIDE; RB1's load: three chats that keep every word, a
        // pass of 12 rows at 365.2 ms and 32.1 ms outside it, plain 72.6 ms): it counts 4.5 periods, half a token a chat.
        let mut stall=Group::new(3,72.4,72.6);stall.rest(72.4,72.6);stall.until_hot(&[SURE,SURE,SURE],&[3,3,3],72.4,72.6);
        let before=stall.pair.behind[0];
        for (tick,(pass,outside)) in [(365.2,32.1),(204.0,32.0),(204.0,32.0)].into_iter().enumerate() {
            for id in 1..=3 {assert_eq!(stall.pair.ask(id,3,3,stall.t,true),3,"tick {tick}, chat {id}");assert_eq!(stall.pair.choose(id,3,SURE,10.0),3,"tick {tick}, chat {id}");}
            assert!(stall.pair.step_begins(true,stall.now));
            if tick==1 {assert!((stall.pair.behind[0]-before-0.5).abs()<1e-9,"one stall costs half a token: {} {before}",stall.pair.behind[0]);}
            stall.pair.passed(3,12,pass);for id in 1..=3 {stall.pair.turned(id,3,3,4,pass/3.0);}stall.now+=pass+outside;stall.t+=1;}
        assert!(stall.pair.hot && stall.pair.account.starts==1,"one stall ended the drafting");
        // Stalls that go on still end the drafting: their losses add up in THE NET, and two of them make a price.
        let mut stalls=0;
        while stall.pair.hot && stalls<8 {let asked=(1..=3).map(|id|stall.pair.ask(id,3,3,stall.t,true)).collect::<Vec<_>>();if asked.contains(&0) {break;}
            for id in 1..=3 {stall.pair.choose(id,3,SURE,10.0);}assert!(stall.pair.step_begins(true,stall.now));
            stall.pair.passed(3,12,365.2);for id in 1..=3 {stall.pair.turned(id,3,3,4,365.2/3.0);}stall.now+=365.2+32.1;stall.t+=1;stalls+=1;}
        assert!(!stall.pair.hot && (2..=5).contains(&stalls),"stalls that go on end the drafting: {stalls}");
        // One slow pass after the first two does not refuse its row count: it is no price until a second pass confirms
        // it, and as the two do not agree within SLOW the lower one is the price.
        let mut once=Pair::default();once.join(1);once.join(2);once.warm=PASS_SKIP;
        once.passed(2,8,400.0);assert_eq!((once.pass[8],once.once[8]),(None,Some(400.0)));once.passed(2,8,130.0);assert_eq!((once.pass[8],once.once[8]),(Some(130.0),None));
        once.passed(2,8,130.0);assert_eq!(once.pass[8],Some(130.0));
        // A pass far above the price is held back: the next one is as the price (it was a stall, dropped), or as slow
        // (then both count).
        let mut held=Pair::default();held.join(1);held.join(2);held.warm=PASS_SKIP;held.pass[12]=Some(207.0);
        held.passed(2,12,308.7);assert_eq!((held.pass[12],held.once[12]),(Some(207.0),Some(308.7)));
        held.passed(2,12,239.5);assert_eq!((held.pass[12],held.once[12]),(Some(0.75*207.0+0.25*239.5),None));
        held.pass[12]=Some(207.0);held.passed(2,12,300.0);held.passed(2,12,300.0);assert_eq!(held.pass[12],Some(0.5*207.0+0.5*300.0));
        // The first price of a row count: two passes that do not agree within SLOW give the lower one.
        let mut first=Pair::default();first.join(1);first.join(2);first.join(3);first.warm=PASS_SKIP;
        first.passed(3,12,206.1);first.passed(3,12,285.4);assert_eq!(first.pass[12],Some(206.1));
        first.pass[12]=None;first.passed(3,12,285.4);first.passed(3,12,206.1);assert_eq!(first.pass[12],Some(206.1));
        first.pass[12]=None;first.passed(3,12,206.0);first.passed(3,12,214.0);assert_eq!(first.pass[12],Some(210.0));
        // The first chat brought its rows, and the second chat's words do not pass: it checks the two words that are
        // worth more than the 11 ms a row they add (it keeps none of them here), and the drafting ends with that tick.
        let mut uneven=primed(0.97);assert_eq!(uneven.tick(&[SURE,PROSE],&[3,0],64.0,64.5),(vec![3,3],vec![3,2]));
        assert!(!uneven.pair.hot && uneven.pair.account.dry_ticks==0 && uneven.pair.passed[..2]==[true,false]);
        // The second chat must pass a look again, and no look is free after a drafting that its own words ended: it
        // lost a token to that one tick, so its next look waits until THE NET has room.
        assert_eq!(uneven.tick(&[SURE,PROSE],&[3,0],64.0,64.5).0,[0,0]);assert!(uneven.pair.behind[1]>0.5 && uneven.pair.behind[0]<0.0 && uneven.pair.doubt[..2]==[false,true],"{:?}",uneven.pair.behind);
        assert!(uneven.rest(64.0,64.5)>100);
        // Then it is sure and right twice in a row (its own words ended the drafting), and the group drafts again with
        // no look of the first chat.
        assert_eq!(uneven.until_hot(&[SURE,SURE],&[3,3],64.0,64.5),[[0,3],[0,0],[0,0],[0,3],[0,0],[0,0],[0,0]]);
        assert_eq!(uneven.tick(&[SURE,SURE],&[3,3],64.0,64.5),(vec![3,3],vec![3,3]));assert_eq!(uneven.pair.account.starts,1);
        // A chat whose look failed by its price and that is sure later must be right twice in a row: one right look
        // may be luck, and a drafting that starts and ends costs every chat. Its second look follows the first at once.
        let mut doubt=Group::new(2,64.0,64.5);doubt.rest(64.0,64.5);
        assert_eq!(doubt.tick(&[PROSE,SURE],&[3,3],64.0,64.5).0,[3,0]);assert!(doubt.pair.again[0]>=2.0*38.0 && doubt.pair.again[0]<80.0,"{:?}",doubt.pair.again);
        while doubt.pair.paced<doubt.pair.again[0] {assert_eq!(doubt.tick(&[SURE,SURE],&[3,3],64.0,64.5).0,[0,0],"the chat waits");}
        doubt.rest(64.0,64.5);assert_eq!(doubt.until_hot(&[SURE,SURE],&[3,3],64.0,64.5),[[3,0],[0,0],[0,0],[3,0],[0,0],[0,0],[0,3],[0,0],[0,0],[0,0]]);
        // A chat whose FIRST look fails in the tick its answer text begins has been in it no time: it still waits
        // (TEXT_SKIP periods at the least) and it must be right twice. (Before, `again` was `paced` and one right look
        // of it started the drafting: MEASURED in every code pair of run draftm-order-o-1, look at step 55.)
        let mut late=Group::new(2,64.0,64.5);late.reads[1]=false;late.rest(64.0,64.5);
        assert_eq!(late.tick(&[SURE,PROSE],&[3,3],64.0,64.5).0,[3,0]);for _ in 0..3 {late.tick(&[SURE,PROSE],&[3,3],64.0,64.5);}
        assert_eq!(late.pair.passed[..2],[true,false]);late.reads=[true;PAIR_CHATS];
        assert_eq!(late.tick(&[SURE,PROSE],&[3,3],64.0,64.5).0,[0,3],"the second chat looks in the tick it begins to read");
        assert!(late.pair.doubt[1] && late.pair.again[1]>=late.pair.paced+f64::from(TEXT_SKIP)-2.0,"{:?} {}",late.pair.again,late.pair.paced);
        // Two chats whose words both fail in one tick: both must pass a look again.
        assert_eq!(weak.pair.passed[..2],[false,false]);
        // Three chats: the looks go round, one chat after the other. A look that fails is its own chat's to pass again.
        let mut three=Group::new(3,75.0,76.0);three.rest(75.0,76.0);
        let first=(0..7).map(|_|three.tick(&[SURE,SURE,PROSE],&[3,3,3],75.0,76.0).0).collect::<Vec<_>>();
        assert_eq!(first,[[3,0,0],[0,0,0],[0,0,0],[0,3,0],[0,0,0],[0,0,0],[0,0,3]]);assert_eq!((&three.pair.passed[..4],three.pair.hot),(&[true,true,false,false][..],false));
        three.rest(75.0,76.0);
        assert_eq!(three.until_hot(&[SURE,SURE,SURE],&[3,3,3],75.0,76.0),[[0,0,3],[0,0,0],[0,0,0],[0,0,3],[0,0,0],[0,0,0],[0,0,0]],"only the chat whose look failed looks again, twice: it failed by its price");
        assert_eq!(three.tick(&[SURE,SURE,SURE],&[3,3,3],75.0,76.0),(vec![3,3,3],vec![3,3,3]));
    }
    #[test] fn the_net_ends_the_drafting_after_one_tick_that_leaves_a_chat_behind() {
        // Both looks pass. Then the prices say that both chats gain while chat 2 keeps nothing.
        let mut group=Group::new(2,64.0,64.5);group.rest(64.0,64.5);group.until_hot(&[SURE,SURE],&[3,3],64.0,64.5);
        assert_eq!(group.tick(&[SURE,SURE],&[3,0],64.0,64.5),(vec![3,3],vec![3,3]));
        // That one tick of 8 rows took two plain periods and gave chat 2 one token: no second one.
        for tick in 0..100 {assert_eq!(group.tick(&[SURE,SURE],&[3,0],64.0,64.5).0,[0,0],"tick {tick}");}
        let (pair,account)=(&group.pair,group.pair.account);assert_eq!((account.starts,account.turn_ticks,account.looks,pair.hot),(1,1,2,false));
        // Chat 2 lost under two tokens in all: the looks, the proposals and that tick. Chat 1 is ahead.
        assert!((1.5..2.0).contains(&pair.behind[1]) && pair.behind[0]<0.0,"{:?}",pair.behind);
        // A stall of the whole service (one tick of two seconds) is not counted as more than STALL plain periods.
        let mut group=Group::new(2,64.0,64.5);group.rest(64.0,64.5);group.until_hot(&[SURE,SURE],&[3,3],64.0,64.5);group.now+=2000.0;
        for _ in 0..3 {assert_eq!(group.tick(&[SURE,SURE],&[3,3],64.0,64.5).1,[3,3]);}
        assert!(group.pair.hot && group.pair.behind[0]<=STALL,"{:?}",group.pair.behind);
    }
    #[test] fn rb1s_flipped_loads_replayed_one_stall_goes_on_and_two_slow_passes_are_the_price() {
        // A REPLAY of the three load-runs in which RB1's bracket rb1-g16 saw the group decision flip (rank-0 lines, logged
        // 2026-11-17). Each plain tick takes the logged plain period; the first drafting tick's proposals take the three
        // chats' logged mean proposal times, and they land in the tick before it, after the drafting began (`floor`).
        // (1) rb1-g16-b1, s3d-D: step 72.4 ms, plain 72.6; proposals 10.5, 10.6 and 10.8 ms; then ONE pass of 12 rows at
        // 365.2 ms in which every chat kept 4. The open tick counts 365.2/72.6 = 5.03 periods up to the limit: under the
        // old limit (rows a chat + 1 = 5) a chat was 1.44 tokens behind, over ROOM, and the drafting ended there, as
        // logged (turn_ticks=1). With PASS_OUTSIDE (4.5) it is 0.94, and the drafting goes on (a margin of 0.06).
        let (step,plain)=(72.4,72.6);
        let mut b1=Group::new(3,step,plain);b1.rest(step,plain);b1.until_hot(&[SURE,SURE,SURE],&[3,3,3],step,plain);
        let floor=b1.pair.floor;
        for id in 1..=3 {assert_eq!(b1.pair.ask(id,3,3,b1.t,true),3);assert_eq!(b1.pair.choose(id,3,SURE,10.6),3);}
        b1.now+=10.5+10.6+10.8;assert!(b1.pair.step_begins(true,b1.now));
        b1.pair.passed(3,12,365.2);for id in 1..=3 {b1.pair.turned(id,3,3,4,365.2/3.0);}b1.now+=365.2;b1.t+=1;
        let proposals=b1.pair.behind[0]-floor[0];assert!((proposals-31.9/plain).abs()<1e-9,"{proposals}");
        let (worst,old)=(b1.pair.worst(&floor),proposals+(365.2/plain).min(12.0/3.0+1.0)-4.0);
        assert!((worst-(proposals+PASS_OUTSIDE)).abs()<1e-9 && worst<ROOM && old>ROOM,"now {worst}, under the old limit {old}");
        for id in 1..=3 {assert_eq!(b1.pair.ask(id,3,3,b1.t,true),3,"the drafting goes on after one stall, chat {id}");}
        assert!(b1.pair.hot && b1.pair.account.starts==1);
        // (2) and (3): rb1-g16-n1, s3c-D and s3-D, each with its own logged times: the batch step and the plain period,
        // four passes of 12 rows in which every chat kept 4 (the first two not priced: PASS_SKIP), and the time the ticks
        // took outside the passes. The log gives that time two ways: the group's outside_ms (each tick after the first
        // here), and the whole span from the last plain step to the next one (the next step's since_last_seconds). What
        // the span leaves is the first tick's proposals: s3c-D 1208.183 - 931.486 - 4 x 64.9 = 17.1 ms; s3-D 1150.028 -
        // 916.233 - 4 x 54.6 = 15.4 ms. In both, the last two passes are the first timed ones and agree within SLOW, so
        // they ARE the price of 12 rows (265.1 and 259.2 ms): at it three chats that keep every word do not pay, and the
        // drafting ends by its price in the next tick (the logged dry tick). THE NET does not end it, under either limit
        // (now 0.52 and -0.07 tokens behind at the last ask; under the old limit 0.76 and -0.07). PLAINLY: this fix does
        // not keep either load grouping. Two slow passes in a row cannot be told from a row count that costs that much.
        let all:&[f64]=&[1.0,1.0,1.0];
        for (load,step,plain,passes,first,outside) in [("s3c-D",71.8,72.1,[203.953,197.301,253.743,276.489],17.1,64.9),
                                                         ("s3-D",72.0,72.2,[201.510,196.305,248.387,270.031],15.4,54.6)] {
            let mut n1=Group::new(3,step,plain);n1.rest(step,plain);n1.until_hot(&[SURE,SURE,SURE],&[3,3,3],step,plain);
            let floor=n1.pair.floor;
            for (tick,pass) in passes.into_iter().enumerate() {
                for id in 1..=3 {assert_eq!(n1.pair.ask(id,3,3,n1.t,true),3,"{load} tick {tick}");assert_eq!(n1.pair.choose(id,3,all,10.6),3,"{load} tick {tick}");}
                n1.now+=if tick==0 {first} else {outside};assert!(n1.pair.step_begins(true,n1.now));
                n1.pair.passed(3,12,pass);for id in 1..=3 {n1.pair.turned(id,3,3,4,pass/3.0);}n1.now+=pass;n1.t+=1;}
            assert_eq!(n1.pair.pass[12],Some(0.5*passes[2]+0.5*passes[3]),"{load}: the two timed passes agree, so they are the price");
            // The open tick counts (last pass + outside)/plain periods up to the limit; the ticks before it were all
            // under 4.5 periods, so the old limit (5) changes the open tick only.
            let open=(passes[3]+outside)/plain;let worst=n1.pair.worst(&floor);let old=worst-(open.min(4.5)-4.0)+(open.min(5.0)-4.0);
            assert!(worst<ROOM && old<ROOM,"{load}: THE NET does not end it: now {worst}, under the old limit {old}");
            for id in 1..=3 {assert_eq!(n1.pair.ask(id,3,3,n1.t,true),3,"{load}");}
            assert_eq!((1..=3).map(|id|n1.pair.choose(id,3,all,10.6)).collect::<Vec<_>>(),[0,0,0],"{load}: two slow passes that agree are the price");
            n1.now+=outside;n1.pair.step_begins(false,n1.now);assert!(!n1.pair.hot && n1.pair.account.dry_ticks==1,"{load}");
        }
    }
    #[test] fn no_look_and_no_start_when_a_chat_has_too_few_tokens_left() {
        // A request of 256 new tokens whose first proposal has a history of 500: 255 left (its first token is chosen).
        let mut text=Text::new(256);assert_eq!(text.left(500),Some(255));assert_eq!(text.left(700),Some(55));assert_eq!(Text::default().left(9),None);
        // Two sure chats: no look while one of them has fewer than LATE tokens left; with room, the group drafts.
        let mut group=Group::new(2,64.0,64.5);group.rest(64.0,64.5);group.pair.left(1,2,Some(500));group.pair.left(2,2,Some(LATE-1));
        for tick in 0..20 {assert_eq!(group.tick(&[SURE,SURE],&[3,3],64.0,64.5).0,[0,0],"tick {tick}");}
        group.pair.left(2,2,Some(LATE));assert_eq!(group.until_hot(&[SURE,SURE],&[3,3],64.0,64.5).len(),7);
        let mut both=Group::new(2,64.0,64.5);both.rest(64.0,64.5);
        assert_eq!(both.tick(&[SURE,SURE],&[3,3],64.0,64.5).0,[3,0]);for _ in 0..2 {both.tick(&[SURE,SURE],&[3,3],64.0,64.5);}
        // The second chat's look waits for its three tokens; with the last of them it passes, and one chat has too few
        // tokens left by then: no drafting starts.
        assert_eq!(both.tick(&[SURE,SURE],&[3,3],64.0,64.5).0,[0,3]);for _ in 0..2 {both.tick(&[SURE,SURE],&[3,3],64.0,64.5);}
        both.pair.left(1,2,Some(10));both.tick(&[SURE,SURE],&[3,3],64.0,64.5);assert!(both.pair.passed[..2]==[true,true] && !both.pair.hot,"{:?}",both.pair.passed);
    }
    #[test] fn the_net_counts_the_slow_steps_after_a_drafting_and_they_are_no_plain_samples() {
        // MEASURED (run draftm-text-j-1): the four steps after a drafting of the code pair took 98.9, 75.7, 69.4 and
        // 66.1 ms where a plain step takes 63.6. That is the drafting's cost: 0.88 plain token for each chat.
        let mut group=Group::new(2,63.0,63.6);group.rest(63.0,63.6);group.until_hot(&[SURE,SURE],&[3,3],63.0,63.6);
        for _ in 0..3 {assert_eq!(group.tick(&[SURE,SURE],&[3,3],63.0,63.6).1,[3,3]);}
        // The drafting ends: no chat's words pay any more.
        assert_eq!(group.tick(&[PROSE,PROSE],&[0,0],63.0,63.6).1,[0,0]);assert!(!group.pair.hot && group.pair.cool==COOL);
        // That tick is the first of the COOL plain ticks (here at the plain period); the next three are slow, and
        // THE NET counts each at its real time. The ticks after them are plain again and are on pace.
        let (before,plain)=(group.pair.behind,group.pair.plain.unwrap());
        for period in [98.9,75.7,69.4,63.6,63.6] {group.plain(63.0,period);}
        let lost=(98.9+75.7+69.4-3.0*63.6)/63.6;
        assert!(group.pair.cool==0 && (group.pair.behind[0]-before[0]-lost).abs()<0.01 && group.pair.behind[0]==group.pair.behind[1],"{:?} {before:?} {lost}",group.pair.behind);
        assert!((group.pair.plain.unwrap()-plain).abs()<1e-6,"a slow step after a drafting is no sample of the plain period: {:?} {plain}",group.pair.plain);
    }
    #[test] fn with_no_price_a_look_checks_no_word_and_passes_only_when_its_words_come() {
        // DRAFT CONFIDENCE off, or a proposal with no probability: `offer`. No row checks a look's words: like every
        // look, it waits for the tokens its chat decodes next (CLAIM A-7). Chat 1's look is right; chat 2's first
        // look is wrong at its second word, so chat 2 waits and must then be right twice in a row.
        fn look_tick(group:&mut Group,words:[&[u32];2])->([usize;2],[usize;2]) {
            let t=group.t;let asked=[group.pair.ask(1,2,3,t,true),group.pair.ask(2,2,3,t,true)];let mut checked=[0;2];
            for at in 0..2 {if asked[at]>0 {checked[at]=group.pair.offer(at as u64+1,2,asked[at]);if checked[at]==0 {group.pair.shadow(at as u64+1,2,words[at]);}}}
            group.plain(64.0,64.5);(asked,checked)
        }
        let mut group=Group::new(2,64.0,64.5);group.rest(64.0,64.5);
        let t=group.t;assert_eq!(look_tick(&mut group,[&[t+1,t+2,t+3],&[]]),([3,0],[0,0]),"chat 1 looks; no row checks its words");
        assert_eq!(group.pair.checking,Some(0));
        for _ in 0..2 {assert_eq!(look_tick(&mut group,[&[],&[]]),([0,0],[0,0]),"no other look while one waits for its tokens");}
        let t=group.t;assert_eq!(look_tick(&mut group,[&[],&[t+1,u32::MAX,u32::MAX]]),([0,3],[0,0]),"chat 1's words came: it passes, and chat 2 looks");
        assert!(group.pair.passed[0] && !group.pair.passed[1] && group.pair.checking==Some(1));
        for _ in 0..2 {assert_eq!(look_tick(&mut group,[&[],&[]]),([0,0],[0,0]));}
        assert!(!group.pair.passed[1] && group.pair.doubt[1] && group.pair.checking.is_none() && group.pair.again[1]>group.pair.paced && !group.pair.hot,"a wrong word fails the look");
        // Chat 2 waits (`again`), then looks again. Right once after a look that failed is not enough: it looks again at once.
        let mut waited=0;
        loop {let t=group.t;let (asked,checked)=look_tick(&mut group,[&[],&[t+1,t+2,t+3]]);assert_eq!(checked,[0,0]);if asked[1]>0 {break;}waited+=1;assert!(waited<200);}
        for _ in 0..2 {assert_eq!(look_tick(&mut group,[&[],&[]]),([0,0],[0,0]));}
        let t=group.t;assert_eq!(look_tick(&mut group,[&[],&[t+1,t+2,t+3]]),([0,3],[0,0]),"right once after a failed look: it looks again at once");
        assert!(!group.pair.passed[1] && !group.pair.hot);
        for _ in 0..3 {assert_eq!(look_tick(&mut group,[&[],&[]]),([0,0],[0,0]));}
        assert!(group.pair.passed[1] && group.pair.hot && group.pair.account.starts==1 && group.pair.account.looks==4,"{:?}",group.pair.account);
        assert_eq!(group.pair.account.drafted_rows,[0;PAIR_CHATS],"no row checked a look's words");
        // While the group drafts, a proposal with no price is checked whole, and THE NET alone ends the drafting when a
        // chat has lost ROOM tokens to it: here both chats keep their own row only.
        let pair=&mut group.pair;let mut now=group.now;let mut ticks=0;
        loop {
            if (pair.ask(1,2,3,0,true),pair.ask(2,2,3,0,true))!=(3,3) {break;}
            assert_eq!((pair.offer(1,2,3),pair.offer(2,2,3)),(3,3));assert!(pair.step_begins(true,now));
            pair.passed(2,8,130.0);pair.turned(1,2,3,1,60.0);pair.turned(2,2,3,1,60.0);now+=152.0;ticks+=1;assert!(ticks<9);
        }
        assert!(!pair.hot && (1..=2).contains(&ticks),"{ticks} ticks");
    }
    #[test] fn a_chat_that_leaves_closes_the_last_tick_first() {
        // The account a group prints counts its last tick too (CLAIM A-3): in a pass of 3 plain periods both chats kept
        // their own row only, so each is 2 tokens further behind its plain pace.
        let mut group=Group::new(2,64.0,64.5);let now=group.now;let pair=&mut group.pair;let before=pair.behind;
        assert!(pair.step_begins(true,now));pair.passed(2,8,193.5);pair.turned(1,2,3,1,60.0);pair.turned(2,2,3,1,60.0);
        let (_,account)=pair.leave(2,now+193.5).unwrap();
        assert!((0..2).all(|at|(account.behind[at]-before[at]-2.0).abs()<1e-9),"{:?} {before:?}",account.behind);
    }
    #[test] fn the_test_control_keeps_every_chat_proposing() {
        let mut pair=Pair::default();pair.join(7);pair.join(9);pair.set_always(true);
        // With the control no timed period is needed, nothing ends a chat, and a tick that was asked is a pass.
        for tick in 0..30 {
            assert_eq!((pair.ask(7,2,3,0,true),pair.ask(9,2,3,0,true)),(3,3));pair.offer(7,2,3);pair.offer(9,2,3);assert!(pair.step_begins(false,400.0*f64::from(tick)));
            pair.passed(2,8,130.0);pair.turned(7,2,3,1,80.0);pair.turned(9,2,3,1,80.0);
        }
        assert!(!pair.step_begins(true,12000.0),"a step with no ask before it is not a pass of the group");
        assert!(!pair.tracking(&[7,9]) && !pair.chooses(7,2) && pair.member(7,2) && pair.behind==[0.0;PAIR_CHATS]);
        pair.join(11);assert_eq!(pair.ask(7,3,3,0,true),3,"three chats draft together");
        pair.join(12);assert_eq!(pair.ask(7,4,3,0,true),3,"four chats draft together");
        for (idx,id) in (13u64..=16).enumerate() {pair.join(id);assert_eq!(pair.ask(7,5+idx,3,0,true),3,"up to PAIR_CHATS draft together");}
        pair.join(17);assert_eq!(pair.ask(7,PAIR_CHATS+1,3,0,true),0,"PAIR_CHATS+1 chats are more than a group holds: plain rows");
    }
    #[test] fn the_first_lone_decisions_after_a_fast_peer_and_after_a_slow_peer_leave_are_the_same() {
        // CLAIM A-5, the reviewer's test. A chat of depth 3 drafts alone, then beside a peer: its rows run in shared
        // passes, fast beside one peer and slow beside another. The peer leaves. Its own records are those it had when
        // the group began, and its first lone decisions are the same after either peer.
        use super::super::gate::Gate;
        // One cycle at position `at` the way the service runs it: a proposal of 3 words (`propose_model`: begin, then
        // the held records back), its pass of 4 rows in `ms` (shared.rs holds the records first when `shared`), and a
        // verdict that keeps `kept` rows (`keep`: the verdict, then the held records back).
        fn cycle(gate:&mut Gate,held:&mut Held,shared:bool,at:u64,ms:f64,kept:u64) {
            gate.begin();held.back(gate);gate.proposed(11.0,1.0,at,std::time::Instant::now(),3,0);
            if shared {held.hold(gate);}gate.pass(4,ms,false);gate.verdict(at+kept,1.0,kept<4);held.back(gate);
        }
        // A proposal cut to none that rides a shared pass with its own row only: shared.rs holds, the pass, back at once.
        fn ride(gate:&mut Gate,held:&mut Held,shared:bool,at:u64) {
            gate.begin();held.back(gate);gate.proposed(11.0,1.0,at,std::time::Instant::now(),3,0);
            if shared {held.hold(gate);}gate.pass(1,70.0,false);if shared {held.back(gate);}
        }
        // `hold`: the service holds the records (this change); else it does not (before it).
        let run=|peer_ms:f64,hold:bool|{
            let (mut gate,mut held)=(Gate::default(),Held::default());
            for at in 0..3 {cycle(&mut gate,&mut held,false,100+10*at,140.0,2);}
            let alone=(gate.pay,gate.deeper);
            for at in 3..8 {cycle(&mut gate,&mut held,hold,100+10*at,peer_ms,4);}
            for at in 8..10 {ride(&mut gate,&mut held,hold,100+10*at);}
            let records=(gate.pay,gate.deeper);
            let decisions=(0..4).map(|_|{let depth=gate.pay.depth(3,gate.begun(),0,44.67);gate.pay.step();depth}).collect::<Vec<_>>();
            (alone,records,decisions)
        };
        let ((alone,fast,fast_decisions),(_,slow,slow_decisions))=(run(70.0,true),run(200.0,true));
        assert!(fast==alone && slow==alone,"the records a chat had when its group began");
        assert_eq!(fast_decisions,slow_decisions);
        // Before this change the two peers left the chat different records and different first decisions (here: whole
        // proposals after the fast peer, plain steps after the slow one).
        let ((_,fast,fast_decisions),(_,slow,slow_decisions))=(run(70.0,false),run(200.0,false));
        assert!(fast!=slow && fast_decisions!=slow_decisions,"{fast_decisions:?} {slow_decisions:?}");
    }
}
