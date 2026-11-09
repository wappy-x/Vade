//! Exact verification of proposed tokens. Rows come from the prompt-row path, whose
//! bits equal `step` for each row. A committed state is never edited: it is replaced
//! by the result of a pass on tokens that were all kept.
use super::{Compute,Decoder,HeadRows,OpenRows,PredictorState,State,PROPOSER_HIDDEN};
use crate::Result;
/// Kept rows that may ride in front of later passes before a pass commits them alone.
pub const CARRIED_ROWS:usize=8;
/// Largest number of new rows in one pass.
pub const VERIFY_ROWS:usize=16;
/// Kept rows that may wait for the predictor layer's cache. Older ones leave a gap in it.
pub const TRAIL_ROWS:usize=4;
/// Last prompt rows that start the predictor layer's cache of a chat. Each runs the cache half
/// of the layer once before the first proposal, so the bound is also the most a request pays
/// before it has drafted.
pub const PROMPT_TRAIL:usize=256;
/// What a pass of `rows` rows costs a request, in ms, from its own timed plain step and its own
/// timed passes (`timed[n]` for a pass of n rows). One row is a plain step. The timed row counts
/// are joined by straight lines, and past the largest one the line from one row goes on. Before
/// any pass is timed, a pass is priced as one step: a price that is too high would never be
/// corrected, because the chooser would never run the pass that corrects it. No number here is
/// measured anywhere else, and bits never depend on it.
pub fn learned_pass_ms(step:f64,timed:&[Option<f64>;17],rows:usize)->f64 {
    if rows<=1 {return step;}let rows=rows.min(16);let (mut low,mut high)=((1usize,step),None);
    for count in 2..=16 {if let Some(ms)=timed[count] {if count<=rows {low=(count,ms);} else if high.is_none() {high=Some((count,ms));}}}
    if low.0==rows {return low.1;}
    let slope=match high {Some((count,ms))=>(ms-low.1)/(count-low.0) as f64,None if low.0>1=>((low.1-step)/(low.0-1) as f64).max(0.0),None=>0.0};
    low.1+slope*(rows-low.0) as f64
}
/// One chat's committed state, the rows appended after it, and their state when every
/// one of them is still kept.
pub struct Verify<V> {state:State<V>,open:Vec<u32>,ahead:Option<State<V>>,kept:Option<OpenRows<V>>,hidden:Option<[V;2]>,
    /// For a proposer: each kept row the predictor layer has not run yet (its input token and
    /// last hidden state), and the layer's own cache of the rows it has run.
    tracking:bool,trail:Vec<(u32,V)>,trail_cap:usize,predictor:Option<PredictorState<V>>}
impl<V> Verify<V> {
    pub fn new(state:State<V>)->Self {Self {state,open:Vec::new(),ahead:None,kept:None,hidden:None,tracking:false,trail:Vec::new(),trail_cap:TRAIL_ROWS,predictor:None}}
    /// Keep what a proposer needs from now on. Off by default: nothing is held.
    pub fn track_proposer(&mut self) {self.tracking=true;}
    /// Whether a proposal can be made now (some kept row is waiting).
    pub fn can_propose(&self)->bool {!self.trail.is_empty()}
    /// Rows appended so far. `truncate` may lower it to any row not yet committed.
    pub fn position(&self)->u64 {self.state.position+self.open.len() as u64}
    /// The last row no `truncate` can drop.
    pub fn committed(&self)->u64 {self.state.position}
    /// The target's last hidden state of the last kept row, before and after the final norm.
    /// None until a step or a pass has run in this state (a prompt leaves none).
    pub fn hidden(&self)->Option<&[V;2]> {self.hidden.as_ref()}
    /// Whether the last pass still holds what a verdict would release.
    pub fn has_kept(&self)->bool {self.kept.is_some()}
    /// Rows the next pass computes again before its new rows.
    pub fn carried(&self)->usize {if self.ahead.is_some() {0} else {self.open.len()}}
    /// Append rows and return the logits of each. Row `i` is exact when the tokens
    /// before it are the tokens plain decoding chose.
    pub fn rows<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,tokens:&[u32])->Result<Vec<Vec<f32>>> {
        if tokens.is_empty() || tokens.len()>VERIFY_ROWS {return Err("verify row count differs".into());}
        self.kept=None;if let Some(ahead)=self.ahead.take() {self.state=ahead;self.open.clear();}
        if self.open.len()>CARRIED_ROWS {let (_,next)=decoder.prefill_rows(&self.open,&self.state,HeadRows::Last)?;self.state=next;self.open.clear();}
        // One row with nothing carried is a plain step: the same bits at the lower price.
        // It is still verification, so it names no next stage (no CHAIN-SHADOW tail blocks).
        if self.open.is_empty() && tokens.len()==1 {
            let (logits,_,next,hidden)=decoder.step_purpose(tokens[0],&self.state,|_,_|Ok(()),false)?;self.open.push(tokens[0]);self.ahead=Some(next);
            if self.tracking {let [before,after]=hidden;self.push_trail(tokens[0],if PROPOSER_HIDDEN==0 {before} else {after});self.hidden=None;} else {self.hidden=Some(hidden);}
            return Ok(vec![logits]);
        }
        let carried=self.open.len();let mut all=self.open.clone();all.extend_from_slice(tokens);
        let (logits,next,kept)=decoder.prefill_rows_open(&all,&self.state,HeadRows::From(carried))?;
        if logits.len()!=tokens.len() {return Err("verify logits row count differs".into());}
        if self.tracking {for row in carried..all.len() {let [before,after]=decoder.open_hidden(&kept,row)?;self.push_trail(all[row],if PROPOSER_HIDDEN==0 {before} else {after});}}
        else {self.hidden=Some(decoder.open_hidden(&kept,all.len()-1)?);}
        self.open=all;self.ahead=Some(next);self.kept=Some(kept);Ok(logits)
    }
    /// Keep the rows before `position`. Later rows were never committed, so nothing is undone:
    /// the state after the kept rows is built from the last pass (`Decoder::prefix_state`). Where
    /// that pass is gone, the kept rows ride in front of the next pass.
    pub fn truncate<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,position:u64)->Result<()> {
        if position<self.state.position || position>self.position() {return Err("verify truncation is outside the open rows".into());}
        let kept=self.kept.take();if position==self.position() {return Ok(());}
        let rows=(position-self.state.position) as usize;
        // The dropped rows leave the trail too: they were never kept.
        let dropped=(self.position()-position) as usize;let left=self.trail.len().saturating_sub(dropped);self.trail.truncate(left);
        self.hidden=match &kept {Some(kept) if rows>0=>Some(decoder.open_hidden(kept,rows-1)?),_=>None};
        self.ahead=match (self.ahead.take(),kept) {(Some(full),Some(kept)) if rows>0=>Some(decoder.prefix_state(&kept,&self.state,&full,rows)?),_=>None};
        self.open.truncate(rows);Ok(())
    }
    fn push_trail(&mut self,token:u32,hidden:V) {if self.trail.len()>=self.trail_cap {self.trail.remove(0);}self.trail.push((token,hidden));}
    /// Start the trail with the last prompt rows: the prompt's last groups in order, each with
    /// its tokens and its hidden rows from `prefill_rows_hidden`. At most PROMPT_TRAIL rows
    /// wait until the first proposal.
    pub fn seed_trail<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,groups:&[(Vec<u32>,OpenRows<V>)])->Result<()> {
        if !self.tracking || !self.trail.is_empty() {return Ok(());}
        let total=groups.iter().map(|group|group.0.len()).sum::<usize>();let mut skip=total.saturating_sub(PROMPT_TRAIL);
        for (tokens,open) in groups {for row in 0..tokens.len() {if skip>0 {skip-=1;continue;}
            let [before,after]=decoder.open_hidden(open,row)?;self.trail.push((tokens[row],if PROPOSER_HIDDEN==0 {before} else {after}));}}
        self.trail_cap=self.trail.len()+TRAIL_ROWS;Ok(())
    }
    /// Proposals for the tokens after `next`, the token chosen for the next row. The predictor
    /// layer first runs once for each kept row it has not seen (its cache of earlier positions),
    /// then proposes. Empty when no kept row waits. Nothing here can change a kept token.
    /// The second result is the seconds spent on the waiting rows, apart from the proposal.
    pub fn propose<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,next:u32,depth:usize)->Result<(Vec<u32>,f64)> {
        if self.trail.is_empty() || depth==0 {return Ok((Vec::new(),0.0));}
        let started=std::time::Instant::now();let mut cache=self.predictor.take();
        for row in 0..self.trail.len()-1 {cache=Some(decoder.propose_after(&self.trail[row].1,self.trail[row+1].0,0,cache.as_ref())?.1);}
        let waited=started.elapsed().as_secs_f64();
        let (tokens,after)=decoder.propose_after(&self.trail[self.trail.len()-1].1,next,depth,cache.as_ref())?;
        self.predictor=Some(after);self.trail.clear();self.trail_cap=TRAIL_ROWS;Ok((tokens,waited))
    }
    /// The state after every kept row.
    pub fn settle<C:Compute<Value=V>>(mut self,decoder:&Decoder<'_,C>)->Result<State<V>> {
        if let Some(ahead)=self.ahead.take() {return Ok(ahead);}
        if self.open.is_empty() {return Ok(self.state);}
        Ok(decoder.prefill_rows(&self.open,&self.state,HeadRows::Last)?.1)
    }
}
