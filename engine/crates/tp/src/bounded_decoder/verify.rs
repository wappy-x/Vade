//! Exact verification of proposed tokens. Rows come from the prompt-row path, whose
//! bits equal `step` for each row. A committed state is never edited: it is replaced
//! by the result of a pass on tokens that were all kept.
use super::{Compute,Decoder,HeadRows,OpenRows,PredictorState,State,PROPOSER_HIDDEN};
use crate::Result;
/// Kept rows that may ride in front of later passes before a pass commits them alone.
pub const CARRIED_ROWS:usize=8;
/// Largest number of new rows in one pass.
pub const VERIFY_ROWS:usize=32;
/// Kept rows that may wait for the predictor layer's cache. Older ones leave a gap in it. The check pass of a
/// proposal deeper than 3 has more rows than this: such a proposal leaves room for every row of its own pass (`propose`).
pub const TRAIL_ROWS:usize=4;
/// Last prompt rows that start the predictor layer's cache of a chat. Each runs the cache half
/// of the layer once before the first proposal, so the bound is also the most a request pays
/// before it has drafted.
pub const PROMPT_TRAIL:usize=256;
/// Waiting rows from which ONE pass brings the predictor layer's cache up (`Decoder::seed_predictor`): the prompt rows
/// before a chat's first proposal. Fewer rows run the layer a row at a time, as every later cycle does (TRAIL_ROWS at
/// most wait there), so no later cycle changes.
pub const SEED_PASS_ROWS:usize=16;
/// Decoded rows that may wait behind the prompt rows for a chat's FIRST proposal (a trial waits for the answer's own
/// text before its first shadow, trial.rs). They take the seeding pass with the prompt rows, and no prompt row is
/// pushed off the trail for them. After the first proposal TRAIL_ROWS bounds the trail as before.
pub const SEED_ROOM:usize=128;
/// What a pass of `rows` rows costs a request, in ms, from its own timed plain step and its own
/// timed passes (`timed[n]` for a pass of n rows). One row is a plain step. The timed row counts
/// are joined by straight lines, and past the largest one the line from one row goes on. Before
/// any pass is timed, a pass is priced as one step: a price that is too high would never be
/// corrected, because the chooser would never run the pass that corrects it. No number here is
/// measured anywhere else, and bits never depend on it.
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
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
    tracking:bool,trail:Vec<(u32,V)>,trail_cap:usize,predictor:Option<PredictorState<V>>,
    /// The hidden states of the first rows of the trail as ONE value, and their count: the prompt rows, for the pass
    /// that seeds the predictor layer. Gone once a row has left the front of the trail.
    plane:Option<(usize,V)>}
impl<V> Verify<V> {
    pub fn new(state:State<V>)->Self {Self {state,open:Vec::new(),ahead:None,kept:None,hidden:None,tracking:false,trail:Vec::new(),trail_cap:TRAIL_ROWS,predictor:None,plane:None}}
    /// Keep what a proposer needs from now on. Off by default: nothing is held. Before the first proposal up to
    /// SEED_ROOM decoded rows may wait so `seed_trail` can wait for the first proposal too.
    pub fn track_proposer(&mut self) {self.tracking=true;if self.predictor.is_none() && self.trail.is_empty() {self.trail_cap=SEED_ROOM;}}
    /// Whether a proposal can be made now (some kept row is waiting).
    pub fn can_propose(&self)->bool {!self.trail.is_empty()}
    /// Kept rows that wait for the predictor layer. A proposal runs the layer's cache once for each but the last.
    pub fn waiting(&self)->usize {self.trail.len()}
    /// Rows appended so far. `truncate` may lower it to any row not yet committed.
    pub fn position(&self)->u64 {self.state.position+self.open.len() as u64}
    /// The last row no `truncate` can drop.
    pub fn committed(&self)->u64 {self.state.position}
    /// The target's last hidden state of the last kept row, before and after the final norm.
    /// None until a step or a pass has run in this state (a prompt leaves none).
    pub fn hidden(&self)->Option<&[V;2]> {self.hidden.as_ref()}
    #[cfg(test)]
    pub(crate) fn predictor_words(&self)->(Option<&PredictorState<V>>,&[(u32,V)]) {
        (self.predictor.as_ref(),&self.trail)
    }
    /// Whether the last pass still holds what a verdict would release.
    pub fn has_kept(&self)->bool {self.kept.is_some()}
    /// Rows the next pass computes again before its new rows.
    pub fn carried(&self)->usize {if self.ahead.is_some() {0} else {self.open.len()}}
    /// Append rows and return the logits of each. Row `i` is exact when the tokens
    /// before it are the tokens plain decoding chose.
    pub fn rows<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,tokens:&[u32])->Result<Vec<Vec<f32>>> {self.rows_head(decoder,tokens,false,None)}
    /// The same pass for a greedy caller, for 2 rows or more: each new row answers with the ID its head picks, taken
    /// where the logits are, and no logits are read (`Decoder::prefill_rows_open_greedy`). The same state, kept rows
    /// and trail as `rows`. One row with nothing carried is `row_greedy`; one row behind carried rows is `rows`.
    pub fn rows_greedy<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,tokens:&[u32])->Result<Vec<u32>> {
        if tokens.len()<2 {return Err("greedy verify rows are 2 or more".into());}
        let mut ids=Vec::new();self.rows_head(decoder,tokens,false,Some(&mut ids))?;
        if ids.len()!=tokens.len() {return Err("verify greedy row count differs".into());}Ok(ids)
    }
    /// An individual service command may also run with other live requests. Both ranks supply
    /// their current live session count for this pass, including one-row commands with carried rows.
    pub(crate) fn session_rows<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,tokens:&[u32],live:usize,greedy:bool)->Result<spark_serve::Predictions> {
        if greedy {
            if tokens.len()<2 {return Err("greedy verify rows are 2 or more".into());}
            let mut ids=Vec::new();self.rows_head(decoder,tokens,live==1,Some(&mut ids))?;
            Ok(spark_serve::Predictions::Greedy(ids))
        } else {Ok(spark_serve::Predictions::Logits(self.rows_head(decoder,tokens,live==1,None)?))}
    }
    /// `picked`: the new rows answer with their IDs, pushed there, and the result is empty (`rows_greedy`).
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    fn rows_head<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,tokens:&[u32],solo_verify:bool,picked:Option<&mut Vec<u32>>)->Result<Vec<Vec<f32>>> {
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
        let ids=picked.is_some();
        let (logits,next,kept)=decoder.verify_open_rows(&all,&self.state,carried,solo_verify,picked)?;
        if !ids && logits.len()!=tokens.len() {return Err("verify logits row count differs".into());}
        if self.tracking {for row in carried..all.len() {let [before,after]=decoder.open_hidden(&kept,row)?;self.push_trail(all[row],if PROPOSER_HIDDEN==0 {before} else {after});}}
        else {self.hidden=Some(decoder.open_hidden(&kept,all.len()-1)?);}
        self.open=all;self.ahead=Some(next);self.kept=Some(kept);Ok(logits)
    }
    /// SHARED PASS, before: the rows this chat brings to a pass it shares with other chats (`shared_pass`): the rows it
    /// carries, then `tokens`, with the count of carried rows. The state they follow is `shared_state`. What `rows`
    /// does before its own pass.
    fn shared_rows<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,tokens:&[u32])->Result<(Vec<u32>,usize)> {
        if tokens.is_empty() || tokens.len()>VERIFY_ROWS {return Err("verify row count differs".into());}
        self.kept=None;if let Some(ahead)=self.ahead.take() {self.state=ahead;self.open.clear();}
        if self.open.len()>CARRIED_ROWS {let (_,next)=decoder.prefill_rows(&self.open,&self.state,HeadRows::Last)?;self.state=next;self.open.clear();}
        let carried=self.open.len();let mut all=self.open.clone();all.extend_from_slice(tokens);Ok((all,carried))
    }
    /// Rows a shared pass would compute for this chat: the rows it carries and `new` rows.
    pub fn shared_count(&self,new:usize)->usize {new+if self.ahead.is_some() || self.open.len()>CARRIED_ROWS {0} else {self.open.len()}}
    /// SHARED PASS, after: this chat's part of the pass, installed as `rows` installs its own pass.
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    fn shared_done<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,all:Vec<u32>,carried:usize,next:State<V>,kept:OpenRows<V>)->Result<()> {
        if self.tracking {for row in carried..all.len() {let [before,after]=decoder.open_hidden(&kept,row)?;self.push_trail(all[row],if PROPOSER_HIDDEN==0 {before} else {after});}}
        else {self.hidden=Some(decoder.open_hidden(&kept,all.len()-1)?);}
        self.open=all;self.ahead=Some(next);self.kept=Some(kept);Ok(())
    }
    /// Append one row as the plain greedy step: the head picks the token on the device, no logits
    /// are read, and the step is the launch program of plain decoding. The same state and the
    /// same kept hidden row as `rows` with one token. None, with nothing changed, when rows are
    /// carried: those ride in front of a pass (`rows`).
    pub fn row_greedy<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,token:u32)->Result<Option<u32>> {
        if self.carried()>0 {return Ok(None);}
        self.kept=None;if let Some(ahead)=self.ahead.take() {self.state=ahead;self.open.clear();}
        let (best,next,hidden)=decoder.step_greedy_hidden(token,&self.state,|_,_|Ok(()))?;self.open.push(token);self.ahead=Some(next);
        if self.tracking {let [before,after]=hidden;self.push_trail(token,if PROPOSER_HIDDEN==0 {before} else {after});self.hidden=None;} else {self.hidden=Some(hidden);}
        Ok(Some(best))
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
    fn push_trail(&mut self,token:u32,hidden:V) {if self.trail.len()>=self.trail_cap {self.trail.remove(0);self.plane=None;}self.trail.push((token,hidden));}
    /// Start the trail with the last prompt rows: the prompt's last groups in order, each with
    /// its tokens and its hidden rows from `prefill_rows_hidden`. Any decoded rows already on the
    /// trail follow behind them, so a short request that never proposes never touches its prompt tail.
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    pub fn seed_trail<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,groups:&[(Vec<u32>,OpenRows<V>)])->Result<()> {
        if !self.tracking || self.predictor.is_some() || self.plane.is_some() {return Ok(());}
        let decoded=std::mem::take(&mut self.trail);
        let total=groups.iter().map(|group|group.0.len()).sum::<usize>();let mut skip=total.saturating_sub(PROMPT_TRAIL);
        // The same rows as one value, group by group, for the pass that seeds the predictor layer (`propose`).
        let mut planes=Vec::new();
        for (tokens,open) in groups {let first=skip.min(tokens.len());if first<tokens.len() {planes.push(decoder.open_hidden_rows(open,first,tokens.len()-first)?);}
            for row in 0..tokens.len() {if skip>0 {skip-=1;continue;}
            let [before,after]=decoder.open_hidden(open,row)?;self.trail.push((tokens[row],if PROPOSER_HIDDEN==0 {before} else {after}));}}
        self.plane=match planes.len() {0=>None,1=>planes.pop().map(|plane|(self.trail.len(),plane)),_=>Some((self.trail.len(),decoder.compute.join(&planes.iter().collect::<Vec<_>>())?))};
        self.trail_cap=self.trail.len()+SEED_ROOM;for (token,hidden) in decoded {self.push_trail(token,hidden);}Ok(())
    }
    fn catch_up<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,waiting:usize)->Result<(Option<PredictorState<V>>,f64)> {
        let started=std::time::Instant::now();let mut cache=self.predictor.take();
        if waiting>=SEED_PASS_ROWS {
            let hidden=decoder.checkpoint.description().hidden as usize;let tokens=self.trail[1..=waiting].iter().map(|row|row.0).collect::<Vec<_>>();
            // The prompt rows are one value already; rows decoded since then follow it one by one.
            let joined=self.plane.as_ref().map(|(rows,plane)|Ok::<_,String>(((*rows).min(waiting),decoder.compute.range(plane,0,(*rows).min(waiting)*hidden)?))).transpose()?;
            let from=joined.as_ref().map_or(0,|(rows,_)|*rows);
            let mut parts=joined.iter().map(|(_,plane)|plane).collect::<Vec<_>>();parts.extend(self.trail[from..waiting].iter().map(|row|&row.1));
            let rows=if parts.len()==1 {None} else {Some(decoder.compute.join(&parts)?)};
            cache=Some(decoder.seed_predictor(rows.as_ref().unwrap_or(parts[0]),&tokens,cache.as_ref())?);
        } else {for row in 0..waiting {cache=Some(decoder.propose_after(&self.trail[row].1,self.trail[row+1].0,0,cache.as_ref())?.1);}}
        Ok((cache,started.elapsed().as_secs_f64()))
    }
    /// Proposals for the tokens after `next`, the token chosen for the next row. The predictor
    /// layer first runs once for each kept row it has not seen (its cache of earlier positions),
    /// then proposes. Empty when no kept row waits. Nothing here can change a kept token.
    /// The second result is the seconds spent on the waiting rows, apart from the proposal.
    /// SEED_PASS_ROWS waiting rows or more (the prompt rows, before a chat's first proposal) take ONE pass of the
    /// layer's cache half for all of them (`Decoder::seed_predictor`): the caches of the row-by-row form.
    pub fn propose<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,next:u32,depth:usize)->Result<(Vec<u32>,f64)> {
        self.propose_mass(decoder,next,depth,None).map(|(tokens,_,waited)|(tokens,waited))
    }
    /// DRAFT CONFIDENCE: `propose`, with the probability of each proposal under the predictor's row (None where the
    /// backend makes none). The same launches and the same state on every rank: only the controller reads the scores.
    #[allow(clippy::type_complexity)] // proposals, their scores and the catch-up seconds.
    /// `stop`: the chain ends early under this probability (`Decoder::propose_after_mass`); every rank passes the same.
    pub fn propose_mass<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,next:u32,depth:usize,stop:Option<f32>)->Result<(Vec<u32>,Vec<Option<f32>>,f64)> {
        if self.trail.is_empty() || depth==0 {return Ok((Vec::new(),Vec::new(),0.0));}
        let (cache,waited)=self.catch_up(decoder,self.trail.len()-1)?;
        let (tokens,mass,after)=decoder.propose_after_mass(&self.trail[self.trail.len()-1].1,next,depth,cache.as_ref(),stop)?;
        // The pass that checks this proposal has up to `depth`+1 rows, and each kept one waits here for the next
        // proposal. With less room a deep pass would push its own first rows off the trail and leave a gap in the
        // layer's cache at every cycle. Up to depth 3 this is TRAIL_ROWS, as before. Both ranks get the same depth.
        self.predictor=Some(after);self.trail.clear();self.trail_cap=TRAIL_ROWS.max(depth+1);self.plane=None;Ok((tokens,mass,waited))
    }
    /// Check what a proposal of `depth` tokens `depth` rows back on the trail would have matched against the kept
    /// tokens that followed it (ending at `best`), stopping the chain on the first wrong token. The rows before that
    /// position take the same `catch_up` pass as `propose` (bit-for-bit the row-by-row form), and the `depth` rows
    /// after it stay on the trail for the next probe or the first real proposal.
    pub fn probe<C:Compute<Value=V>>(&mut self,decoder:&Decoder<'_,C>,best:u32,depth:usize)->Result<(usize,f64)> {
        if depth==0 || self.trail.len()<=depth {return Ok((0,0.0));}
        let at=self.trail.len()-1-depth;let (cache,waited)=self.catch_up(decoder,at)?;
        let mut expected=self.trail[at+2..].iter().map(|row|row.0).collect::<Vec<_>>();expected.push(best);
        let (matched,after)=decoder.propose_until(&self.trail[at].1,self.trail[at+1].0,&expected,cache.as_ref())?;
        self.predictor=Some(after);self.trail.drain(..=at);self.trail_cap=TRAIL_ROWS+depth;self.plane=None;Ok((matched,waited))
    }
    /// Commit the latest complete pass while retaining the predictor cache and trail for a shared step.
    /// A partial rollback must first run its carried rows through the ordinary verify path.
    pub fn batch_state(&mut self)->Result<&State<V>> {
        if self.carried()>0 {return Err("tracked batch step has carried rows".into());}
        if let Some(ahead)=self.ahead.take() {self.state=ahead;self.open.clear();}
        self.kept=None;Ok(&self.state)
    }
    /// The committed state after `batch_state`, for the shared decoder's immutable row table.
    pub fn batch_state_ref(&self)->Result<&State<V>> {
        if !self.open.is_empty() || self.ahead.is_some() {return Err("tracked batch state is not committed".into());}Ok(&self.state)
    }
    /// Install one exact shared-step result. It is committed, as an ordinary plain batch row is.
    pub fn batch_advanced(&mut self,token:u32,next:State<V>,hidden:V)->Result<()> {
        if !self.open.is_empty() || next.position!=self.state.position+1 {return Err("tracked batch position differs".into());}
        self.state=next;self.hidden=None;
        if self.tracking {self.push_trail(token,hidden);}Ok(())
    }
    /// The state after every kept row.
    pub fn settle<C:Compute<Value=V>>(mut self,decoder:&Decoder<'_,C>)->Result<State<V>> {
        if let Some(ahead)=self.ahead.take() {return Ok(ahead);}
        if self.open.is_empty() {return Ok(self.state);}
        Ok(decoder.prefill_rows(&self.open,&self.state,HeadRows::Last)?.1)
    }
}
/// ONE PASS: `Verify::propose_mass` of several chats, `chats[i]` = (its rows, its `next`, its depth, its stop). Each
/// chat's waiting rows run first, as there; then the chains run word by word together (`Decoder::propose_chains`: each
/// chat's steps on its own values, one head pass a word). Each chat's proposals, scores, cache and trail are what its
/// own `propose_mass` leaves. `decoder` is any chat's: the chats of one service share its weights and its plans.
#[allow(clippy::type_complexity)] // each chat's proposals, their scores and its catch-up seconds.
pub fn propose_group<C:Compute>(decoder:&Decoder<'_,C>,chats:&mut [(&mut Verify<C::Value>,u32,usize,Option<f32>)])->Result<Vec<(Vec<u32>,Vec<Option<f32>>,f64)>> {
    let mut caught=Vec::with_capacity(chats.len());
    for (verify,_,depth,_) in chats.iter_mut() {
        caught.push(if verify.trail.is_empty() || *depth==0 {None} else {Some(verify.catch_up(decoder,verify.trail.len()-1)?)});
    }
    let mut made={let starts=chats.iter().zip(&caught).filter_map(|((verify,next,depth,stop),caught)|caught.as_ref().map(|(cache,_)|
        super::ChainStart {hidden:&verify.trail[verify.trail.len()-1].1,token:*next,depth:*depth,seen:cache.as_ref(),stop:*stop})).collect::<Vec<_>>();
        decoder.propose_chains(&starts)?.into_iter()};
    let mut results=Vec::with_capacity(chats.len());
    for ((verify,_,depth,_),caught) in chats.iter_mut().zip(caught) {
        let Some((_,waited))=caught else {results.push((Vec::new(),Vec::new(),0.0));continue;};
        let (tokens,mass,after)=made.next().ok_or("proposal group lost a chain")?;
        verify.predictor=Some(after);verify.trail.clear();verify.trail_cap=TRAIL_ROWS.max(*depth+1);verify.plane=None;results.push((tokens,mass,waited));
    }
    Ok(results)
}
/// SHARED PASS: ONE pass appends `tokens` to each chat (`Decoder::prefill_rows_shared`). Result `i` is the logits of
/// chat `i`'s new rows, with the bits of its own `Verify::rows`; each chat's open rows, its state ahead and its trail
/// are what its own pass leaves, so each takes its own `truncate` afterwards. `decoder` is any chat's: the chats of
/// one service share its weights and its plans.
pub fn shared_pass<C:Compute>(decoder:&Decoder<'_,C>,chats:&mut [(&mut Verify<C::Value>,&[u32])])->Result<Vec<Vec<Vec<f32>>>> {
    let mut rows=Vec::with_capacity(chats.len());
    for (verify,tokens) in chats.iter_mut() {rows.push(verify.shared_rows(decoder,tokens)?);}
    let results={let shared=chats.iter().zip(&rows).map(|((verify,_),(all,carried))|super::SharedRows {tokens:all,state:&verify.state,head:*carried}).collect::<Vec<_>>();
        decoder.prefill_rows_shared(&shared)?};
    let mut logits=Vec::with_capacity(chats.len());
    for (((verify,tokens),(all,carried)),(own,next,kept)) in chats.iter_mut().zip(rows).zip(results) {
        if own.len()!=tokens.len() {return Err("shared pass logits row count differs".into());}
        verify.shared_done(decoder,all,carried,next,kept)?;logits.push(own);
    }
    Ok(logits)
}
/// Greedy form of `shared_pass`: each chat's new rows answer with the token ID picked on the device
/// (`Decoder::prefill_rows_shared_greedy`), avoiding full-vocabulary D2H logits readbacks.
pub fn shared_pass_greedy<C:Compute>(decoder:&Decoder<'_,C>,chats:&mut [(&mut Verify<C::Value>,&[u32])])->Result<Vec<Vec<u32>>> {
    let mut rows=Vec::with_capacity(chats.len());
    for (verify,tokens) in chats.iter_mut() {rows.push(verify.shared_rows(decoder,tokens)?);}
    let results={let shared=chats.iter().zip(&rows).map(|((verify,_),(all,carried))|super::SharedRows {tokens:all,state:&verify.state,head:*carried}).collect::<Vec<_>>();
        decoder.prefill_rows_shared_greedy(&shared)?};
    let mut ids=Vec::with_capacity(chats.len());
    for (((verify,tokens),(all,carried)),(own,next,kept)) in chats.iter_mut().zip(rows).zip(results) {
        if own.len()!=tokens.len() {return Err("shared pass greedy row count differs".into());}
        verify.shared_done(decoder,all,carried,next,kept)?;ids.push(own);
    }
    Ok(ids)
}
