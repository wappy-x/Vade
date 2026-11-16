//! How deep a drafting request proposes when its depth is above the base depth (TRIAL_DEPTH): as deep as its own
//! proposals are kept.
//!
//! A deeper proposal costs one more step of the predictor layer and one more row of the check pass. It gives one more
//! token only when it is kept itself, with every proposal before it. Two Sparks, 2026-11-15: a proposal is 3.4 ms and
//! a row of a pass above four rows about 13 ms, where a drafting request pays 25 to 29 ms for a token. So a deeper
//! proposal costs about two thirds of a token. Structured text keeps nearly every proposal up to depth 5 (35 of 40
//! cycles kept all five), and its sixth and seventh proposals are lost in every second cycle. Code keeps all three of
//! depth 3 in half of its cycles, and a deeper proposal after such a cycle is lost four times of five. Prose keeps
//! all three in a quarter of its cycles. So the depth of each cycle follows the cycles before it:
//! 1. The depths a request uses are LEVELS: the base depth, then DEEPER_STEP deeper each, and the request's whole
//!    depth as the last. A request starts at the base depth.
//! 2. Each level above the base keeps a SCORE in thirds of a token. A cycle at that depth or deeper adds KEPT for each
//!    proposal it kept above the level under it, and takes COST for each proposal the level makes above that level.
//!    With a step of 2: a whole keep +2, one proposal short -1, nothing gained -4. The score holds at most BANK.
//! 3. A cycle that kept its whole proposal goes one level deeper the next time, when that level does not wait.
//! 4. A level whose score comes down to LEAVE is left: the next cycle is at the level under it, and the level waits
//!    WAIT_FIRST cycles before a whole keep can try it again, with a score of 0. Each time it is left again the wait
//!    is WAIT_GROWTH times longer, up to WAIT_MOST. A score at BANK puts the wait back to WAIT_FIRST.
//!
//! So one whole keep is enough to try a deeper level, a level that gives nothing is left after one cycle, and a level
//! that never pays is tried in fewer and fewer cycles: 4 times in the first 100 cycles of a request.
//!
//! A cycle whose proposal was cut before its pass (a stop token, the end of the answer) says nothing and changes
//! nothing. A request whose whole depth is the base depth or less never moves: it proposes its whole depth, as before
//! this file. The rule holds no time: the same kept counts give the same depths every time. It runs on the controller
//! only, and the depth it picks rides in the proposal command, so both ranks run the same chain. Print and decision
//! input only: every proposed row is still verified, and no kept token depends on it.
/// The depth a request starts at, and never goes under: the depth of a trial probe.
pub(super) const BASE:usize=super::trial::TRIAL_DEPTH;
/// Proposals between two levels. Two: a pass of 6 or 8 rows costs little more than one of 5 or 7 (two Sparks,
/// 2026-11-15, median ms by rows: 4 rows 102.8, 5 rows 125.4, 6 rows 130.2, 7 rows 147.5, 8 rows 157.9).
pub(super) const DEEPER_STEP:usize=2;
/// The score counts thirds of a token: what a kept proposal above the level under gives, and what a proposal costs.
pub(super) const KEPT:i32=3;
pub(super) const COST:i32=2;
/// The most a level's score holds, and the score at which a level is left.
pub(super) const BANK:i32=8;
pub(super) const LEAVE:i32=-4;
/// Counted cycles a level waits after it is left: the first time, the factor for each time after it as a shift
/// (4 times), and the most times the factor is taken (2, 8, 32, 128 cycles).
pub(super) const WAIT_FIRST:u32=2;
const WAIT_GROWTH_SHIFT:u32=2;
const WAIT_GROWTHS:u32=3;
/// The longest wait.
pub(super) const WAIT_MOST:u32=WAIT_FIRST<<(WAIT_GROWTH_SHIFT*WAIT_GROWTHS);
/// Depths the account prints: 0 to the largest predictor depth.
const DEPTHS:usize=spark_engine::draft::PREDICTOR_DEPTH_MAX+1;
/// One level above the base depth.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
struct Level {
    /// What the cycles at this depth or deeper gained over the level under it, less what they cost. At most BANK.
    score:i32,
    /// Counted cycles under this level before a whole keep can try it again.
    wait:u32,
    /// Times this level was left since its score was last at BANK, at most WAIT_GROWTHS.
    left:u32}
/// One request's depth above the base depth, its levels, and its account by depth.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub(super) struct Deeper {
    /// Proposals above the base depth that the next cycle makes.
    above:usize,
    /// The request's whole depth, as the last call of `depth` gave it.
    whole:usize,
    /// The levels, by their depth. The places of depths that are no level stay unused.
    levels:[Level;DEPTHS],
    /// For the print, by the depth a cycle ran at: counted cycles, and those that kept their whole proposal.
    cycles:[u32;DEPTHS],kept_whole:[u32;DEPTHS]}
impl Deeper {
    /// The depth of the next proposal of a request whose whole depth is `whole`.
    pub(super) fn depth(&mut self,whole:usize)->usize {self.whole=whole;self.now()}
    fn now(&self)->usize {if self.whole<=BASE {self.whole} else {(BASE+self.above).min(self.whole)}}
    /// The level under the level at `depth`, for a depth above the base.
    fn under(depth:usize)->usize {BASE+(depth-BASE-1)/DEEPER_STEP*DEEPER_STEP}
    /// A cycle ended: its pass verified `drafted` proposed rows and kept `kept` rows, the first row included.
    pub(super) fn verdict(&mut self,drafted:u64,kept:u64) {
        let now=self.now();
        // A request at or under the base depth has nothing to choose. A cut proposal says nothing about this depth.
        if self.whole<=BASE || drafted!=now as u64 || now>=DEPTHS {return;}
        let kept=(kept.saturating_sub(1) as usize).min(now);
        self.cycles[now]+=1;if kept==now {self.kept_whole[now]+=1;}
        // This cycle tried every level from its depth down to the base. `stay`: the level under the lowest level left.
        let (mut level,mut stay)=(now,now);
        while level>BASE {
            let under=Self::under(level);let gained=(kept.min(level)-kept.min(under)) as i32;let tried=&mut self.levels[level];
            tried.score=(tried.score+KEPT*gained-COST*(level-under) as i32).min(BANK);
            if tried.score>=BANK {tried.left=0;}
            if tried.score<=LEAVE {*tried=Level {score:0,wait:(WAIT_FIRST<<(WAIT_GROWTH_SHIFT*tried.left)).min(WAIT_MOST),left:(tried.left+1).min(WAIT_GROWTHS)};stay=under;}
            level=under;
        }
        // A cycle under a level is one cycle of that level's wait.
        for waiting in &mut self.levels[now+1..] {waiting.wait=waiting.wait.saturating_sub(1);}
        let deeper=(now+DEEPER_STEP).min(self.whole);
        let next=if stay<now {stay} else if kept==now && self.levels[deeper].wait==0 {deeper} else {now};
        self.above=next-BASE;
    }
    /// One line for a request whose whole depth is above the base depth: its cycles by depth, and its levels.
    pub(super) fn report(&self,rank:u32,session:u64) {
        if self.whole<=BASE {return;}
        let by=|counts:&[u32;DEPTHS]|(BASE..DEPTHS.min(self.whole+1)).map(|depth|format!("{depth}:{}",counts[depth])).collect::<Vec<_>>().join(",");
        let levels=(BASE+1..DEPTHS.min(self.whole+1)).filter(|&depth|depth==self.whole || (depth-BASE).is_multiple_of(DEEPER_STEP))
            .map(|depth|{let level=self.levels[depth];format!("{depth}:{}/{}/{}",level.score,level.wait,level.left)}).collect::<Vec<_>>().join(",");
        eprintln!("SERVED DEEPER rank={rank} session={session} whole_depth={} base_depth={BASE} step={DEEPER_STEP} cycles_by_depth={} kept_whole_by_depth={} next_depth={} level_score_wait_left={levels}",
            self.whole,by(&self.cycles),by(&self.kept_whole),self.now());
    }
}
#[cfg(test)] mod tests {
    use super::*;
    /// The depths of a request of whole depth `whole` whose cycles keep `kept(i)` proposals of what they made.
    fn depths(whole:usize,kept:&[usize])->Vec<usize> {
        let mut deeper=Deeper::default();
        kept.iter().map(|&kept|{let depth=deeper.depth(whole);deeper.verdict(depth as u64,1+kept.min(depth) as u64);depth}).collect()
    }
    #[test] fn a_request_at_or_under_the_base_depth_always_proposes_its_whole_depth() {
        assert_eq!(BASE,3);
        for whole in 0..=BASE {
            let mut deeper=Deeper::default();
            for kept in [whole,0,whole,whole,1,0,whole,whole,whole] {assert_eq!(deeper.depth(whole),whole);deeper.verdict(whole as u64,1+kept.min(whole) as u64);}
            assert_eq!(deeper,Deeper {whole,..Deeper::default()},"nothing is counted and nothing moves at depth {whole}");
        }
    }
    #[test] fn whole_keeps_go_one_level_deeper_up_to_the_whole_depth() {
        assert_eq!(depths(7,&[9;6]),[3,5,7,7,7,7]);assert_eq!(depths(5,&[9;4]),[3,5,5,5]);assert_eq!(depths(4,&[9;3]),[3,4,4]);assert_eq!(depths(6,&[9;4]),[3,5,6,6]);
        // The largest depth this build takes.
        assert_eq!(*depths(spark_engine::draft::PREDICTOR_DEPTH_MAX,&[9;8]).last().unwrap(),spark_engine::draft::PREDICTOR_DEPTH_MAX);
        // The level under a depth: the step under it, and for a whole depth between two levels the level before it.
        assert_eq!([4,5,6,7].map(Deeper::under),[3,3,5,5]);
    }
    #[test] fn a_request_that_keeps_no_whole_proposal_stays_at_the_base() {
        // Prose: one proposal kept of three. The depth never leaves the base, and no level is scored.
        assert_eq!(depths(7,&[1,0,1,2,1,0]),[3;6]);
        let mut deeper=Deeper::default();for kept in [1,0,2,2,1] {assert_eq!(deeper.depth(7),3);deeper.verdict(3,1+kept);}
        assert_eq!(deeper.levels,[Level::default();DEPTHS]);assert_eq!(deeper.cycles[3],5);assert_eq!(deeper.kept_whole[3],0);
    }
    #[test] fn a_level_that_gives_nothing_is_left_at_once_and_waits_longer_each_time() {
        assert_eq!((WAIT_FIRST,WAIT_MOST),(2,128));
        // Code: all three kept at the base, and nothing above them. The level at 5 is tried after 2, then 8, then 32 cycles.
        let found=depths(7,&[3;60]);
        let tries=found.iter().enumerate().filter(|(_,depth)|**depth==5).map(|(at,_)|at).collect::<Vec<_>>();
        assert_eq!(tries,[1,4,13,46]);assert!(found.iter().all(|depth|*depth==3 || *depth==5),"the level at 7 is never reached");
        // The wait grows to WAIT_MOST and no more.
        let found=depths(7,&[3;500]);
        let tries=found.iter().enumerate().filter(|(_,depth)|**depth==5).map(|(at,_)|at).collect::<Vec<_>>();
        assert_eq!(tries,[1,4,13,46,175,304,433]);
        // A cycle without a whole keep is still a cycle of the wait, and the try waits for the next whole keep.
        assert_eq!(depths(7,&[3,0, 1,2,2,3, 5]),[3,5, 3,3,3,3, 5]);
    }
    #[test] fn a_level_holds_while_its_kept_proposals_pay_for_it() {
        // Step 2: a whole keep scores 2, one proposal short -1, nothing gained -4.
        let mut deeper=Deeper::default();assert_eq!(deeper.depth(5),3);deeper.verdict(3,4);
        for (kept,score) in [(5,2),(4,1),(5,3),(3,-1),(5,1),(5,3),(5,5),(5,7),(5,BANK),(5,BANK),(2,4),(0,0)] {
            assert_eq!(deeper.depth(5),5);deeper.verdict(5,1+kept);assert_eq!(deeper.levels[5].score,score,"kept {kept}");}
        // The third cycle in a row that gains nothing, after a full bank, leaves the level.
        assert_eq!(deeper.depth(5),5);deeper.verdict(5,1+3);assert_eq!(deeper.depth(5),3);assert_eq!(deeper.levels[5],Level {score:0,wait:WAIT_FIRST,left:1});
        // Structured text of 2026-11-15: at depth 5 nearly every cycle keeps all five.
        assert_eq!(depths(5,&[3,4,5,5,5,5,5,5,4,5,5,5,3,5,5,5,4,5]).iter().filter(|depth|**depth==5).count(),17);
        // One proposal short in every cycle does not pay: four such cycles leave the level.
        assert_eq!(depths(5,&[3,4,4,4,4,3]),[3,5,5,5,5,3]);
        // A whole depth one above a level: a whole keep scores 1, a loss -2. The scores here: 1, -1, 0, -2, -4.
        assert_eq!(depths(4,&[3,4,3,4,3,3,3,3]),[3,4,4,4,4,4,3,3]);
    }
    #[test] fn a_full_bank_puts_the_wait_back_to_the_first_wait() {
        let mut deeper=Deeper::default();
        // Left two times: the next wait would be 32 cycles.
        for (depth,kept) in [(3,3),(5,3),(3,3),(3,3),(5,3)] {assert_eq!(deeper.depth(5),depth);deeper.verdict(depth as u64,1+kept);}
        assert_eq!(deeper.levels[5],Level {score:0,wait:8,left:2});
        for _ in 0..8 {assert_eq!(deeper.depth(5),3);deeper.verdict(3,4);}
        for _ in 0..4 {assert_eq!(deeper.depth(5),5);deeper.verdict(5,6);}
        assert_eq!(deeper.levels[5],Level {score:BANK,wait:0,left:0});
        for _ in 0..3 {assert_eq!(deeper.depth(5),5);deeper.verdict(5,1);}
        assert_eq!(deeper.levels[5],Level {score:0,wait:WAIT_FIRST,left:1});
    }
    #[test] fn a_deep_cycle_is_a_cycle_of_every_level_under_it() {
        // Structured text of 2026-11-15 at whole depth 7: depth 5 holds, and a cycle of 7 keeps 6 or 5.
        let mut deeper=Deeper::default();
        for (depth,kept) in [(3,3),(5,5),(7,6),(7,5)] {assert_eq!(deeper.depth(7),depth);deeper.verdict(depth as u64,1+kept);}
        // The level at 7 scored -1 and -4 and is left. The level at 5 took all four cycles: 2 for each.
        assert_eq!(deeper.depth(7),5);assert_eq!(deeper.levels[7],Level {score:0,wait:WAIT_FIRST,left:1});assert_eq!(deeper.levels[5].score,6);
        // Two cycles at 5, then 7 again.
        assert_eq!(depths(7,&[3,5,6,5,5,5,5,5]),[3,5,7,7,5,5,7,5]);
        // A cycle of 7 that keeps 2 gains nothing at 7 and nothing at 5. Both levels are left when the level at 5 has no bank.
        let mut deeper=Deeper::default();
        for (depth,kept) in [(3,3),(5,4),(5,5),(7,2)] {assert_eq!(deeper.depth(7),depth);deeper.verdict(depth as u64,1+kept);}
        assert_eq!(deeper.depth(7),5);assert_eq!((deeper.levels[5].score,deeper.levels[7].wait),(-3,WAIT_FIRST));
        deeper.verdict(5,1+4);assert_eq!(deeper.depth(7),3);assert_eq!((deeper.levels[5].wait,deeper.levels[7].wait),(WAIT_FIRST,1));
    }
    #[test] fn a_cut_proposal_changes_nothing_and_the_account_counts_by_depth() {
        let mut deeper=Deeper::default();
        assert_eq!(deeper.depth(7),3);deeper.verdict(3,4);assert_eq!(deeper.depth(7),5);
        // The scheduler cut this proposal to 2 rows (a stop token): no count and no move, kept or not.
        let before=deeper;deeper.verdict(2,3);deeper.verdict(2,1);assert_eq!(deeper,before);assert_eq!(deeper.depth(7),5);
        deeper.verdict(5,6);assert_eq!(deeper.depth(7),7);deeper.verdict(7,3);assert_eq!(deeper.depth(7),5);
        assert_eq!((deeper.cycles[3],deeper.cycles[5],deeper.cycles[7]),(1,1,1));assert_eq!((deeper.kept_whole[3],deeper.kept_whole[5],deeper.kept_whole[7]),(1,1,0));
        // A smaller whole depth later bounds the depth at once.
        assert_eq!(deeper.depth(4),4);
    }
    #[test] fn the_same_kept_counts_give_the_same_depths() {
        let kept=[3,5,7,7,1,7,0,0,3,3,5,2,6,7,7,7,4,4,3,3,3,5,5,5,5,5,7,7,7,7,6,6,6,6,2];
        assert_eq!(depths(7,&kept),depths(7,&kept));
        for whole in 4..=spark_engine::draft::PREDICTOR_DEPTH_MAX {for depth in depths(whole,&kept) {
            assert!(depth==whole || (depth>=BASE && depth<whole && (depth-BASE).is_multiple_of(DEEPER_STEP)),"depth {depth} of {whole} is a level");}}
    }
}
