//! The decode driver of the block family with a loop: one token through every layer visit that the config lists.
//! It is written once over the trait `Compute`, so the same code runs on the host and on the device.
//! The host run of this driver is the byte reference of its device run.
//!
//! The parts, one file each:
//! - `state`: the key and value rings and planes of each visit, the convolution rings, the slot rule, a clean state.
//! - `attention`: the attention half of a visit. It takes the mixed stream and gives the stream after the first
//!   short convolution.
//! - `mix`: the block mix, the messages and the gates of a visit that runs inside the loop.
//! - `experts`: the expert half of a visit, behind one function. It takes the stream after attention and gives the
//!   stream at the end of the visit. Only this file knows the stored form of the experts.
//! - `operands`: each weight of a visit by its name.
//! - `fixture`: two small made-up checkpoints. A check program runs the driver on them, on the host and on the
//!   device, before any stored weight is bound.
//! - `host_weights`: the weights of a model folder as host words, for a run of the driver on the host backend.
//! - `rows`: many rows a step. The row list, the limits of a step and the launch of one arm on many rows.
//! - `service`: the engine and the session of the text service over this driver. A prompt in pieces of rows, and
//!   the new tokens of several chats in one step.
//!
//! Every size and every visit comes from the description of the checkpoint. No line here names a model, a layer
//! count or a count of ranks.
pub mod state;
pub mod attention;
pub mod mix;
pub mod experts;
pub mod operands;
pub mod fixture;
pub mod host_weights;
pub mod rows;
pub mod service;

use crate::{Result,bounded_decoder::{Compute,greedy_logits}};
use attention::rounded_norm;
use engine_model::{plan::description::LoopTextDescription,shapes::Ffn};
use mix::{MESSAGES,Run,Token,Weight,Weights,need,number,reciprocal,shape};
use rows::{ChatRows,GATHER_ROWS,Limits,Rows,gather};
use spark_gpu_probe::{causal_ops::Op,prompt_ops};
use state::{Layout,Ring,State};

/// What one token step gives.
#[derive(Debug)]
pub struct Step<V> {
    /// The capped logits of the head rows that the model can write: 32-bit, one word a row.
    pub logits:V,
    /// The row with the largest logit, and the lower row when two logits are equal.
    pub top:u32,
}
/// The decode driver of one checkpoint on one backend. It holds no chat: a chat is a `State`.
/// The visits of a token are those of the description, in its order. Each visit has its own state, at the state
/// slot that the description gives it.
pub struct Driver<'a,C:Compute,W:Weights<C>> {run:Run<'a,C,W>,layouts:Vec<Layout>,capacity:u64,limits:Limits}
/// Which rows of one chat of a step run the end: the end mix, the last norm, the head, the cap and the top row.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Wanted {
    /// No row: a piece of a prompt that is not its last piece.
    None,
    /// The last row of the chat in this step.
    Last,
    /// Every row of the chat in this step.
    All,
}
/// One chat of a step of rows: its state, its next tokens in order (at least one; they are the tokens at
/// `state.position` and after it), the rows that run the end, and if the caller reads the logits of those rows
/// (`false`: only their top rows).
pub struct ChatStep<'s,V> {pub state:&'s mut State<V>,pub tokens:&'s [u32],pub wanted:Wanted,pub logits:bool}
/// What a step of rows gives. The wanted rows are in row order: those of chat 0 of the step, then of chat 1, and so
/// on, in each chat the positions rising. So the counts of `Wanted` tell which row belongs to which chat.
#[derive(Debug)]
pub struct StepRows<V> {
    /// The top row of each wanted row: the largest capped logit, and the lower row when two logits are equal.
    pub tops:Vec<u32>,
    /// The capped logits of the wanted rows of the chats that asked for logits, `[those rows][head rows]`,
    /// row-major, 32-bit. `None` when no chat asked. It lives to the end of the work scope of the caller.
    pub logits:Option<V>,
}
impl<'a,C:Compute,W:Weights<C>> Driver<'a,C,W> {
    /// `capacity` is the longest chat, in tokens, that a state of this driver can hold. A layer that sees every
    /// token keeps one row a token, so the capacity is a size of the state.
    /// This driver takes one row a step: the state sizes are those of `Limits::ONE`.
    pub fn new(compute:&'a C,weights:&'a W,description:&'a LoopTextDescription,capacity:u64)->Result<Self> {
        Self::with_limits(compute,weights,description,capacity,Limits::ONE)
    }
    /// A driver for steps of many rows. `limits.step_rows` is the most rows of one step; it is a size of the state
    /// (see `Limits`), so two drivers with two limits hold a chat in states of two sizes, with the same words.
    pub fn with_limits(compute:&'a C,weights:&'a W,description:&'a LoopTextDescription,capacity:u64,limits:Limits)->Result<Self> {
        let d=description;
        need(d.head_rows>0 && d.head_rows<=d.vocab && d.epsilon>0.0 && d.caps.logits>0.0,"the head rows, the norm epsilon or the logit cap are not built")?;
        need(limits.step_rows>0 && limits.key_bytes>0,"a limit of a step is zero")?;
        // Refused here, before any step: a step row limit that a launch of a step cannot take. One launch of the
        // rows entry of the arms takes `block_rows::ROWS_MAX` rows, one product or gather of rows
        // `prompt_ops::MAX_ROWS` (the same count). A gather names its source rows in 32-bit floats: the rows of a step
        // times the longest block list, the messages of a pass, the picked experts or the taps of a ring.
        let most=d.layers.iter().map(|layer|match layer.ffn {Ffn::Experts(experts)=>experts.selected,_=>0}.max(layer.attention.key_value_conv.kernel+1)).max().unwrap_or(0)
            .max(d.core.end.saturating_sub(d.core.start).saturating_mul(MESSAGES as u64)).max(d.visits.len() as u64+1).max(d.conv.kernel+1);
        need(limits.step_rows<=prompt_ops::MAX_ROWS.min(spark_gpu_probe::block_rows::ROWS_MAX) && most<=prompt_ops::MAX_ROWS as u64,"the step row limit is above the rows of one launch")?;
        need((limits.step_rows as u64).checked_mul(most).is_some_and(|rows|rows<=GATHER_ROWS as u64),"the step row limit is above the rows of one gather")?;
        // The limits go into the layouts: the extra rows of a sliding ring, and the byte limit of the part over keys.
        Ok(Self {run:Run::new(compute,weights,d)?,layouts:state::layouts_for(d,capacity,limits)?,capacity,limits})
    }
    /// The limits of a step of this driver.
    pub fn limits(&self)->Limits {self.limits}
    /// The longest chat that a state of this driver holds.
    pub fn capacity(&self)->u64 {self.capacity}
    /// The 32-bit words of the state of one chat: what `clean` allocates.
    pub fn state_words(&self)->usize {self.layouts.iter().map(Layout::words).sum()}
    /// The sizes of the state of each visit, by state slot.
    pub fn layouts(&self)->&[Layout] {&self.layouts}
    /// An upper bound of the 32-bit words that one step of rows holds at one time in values of its own, besides
    /// the states of its chats (`state_words`). Four times the number is bytes.
    /// `rows` is the count of rows of the step, `chats` the count of its chats, `ends` the count of rows that run
    /// the end (`Wanted`) and `logits` the count of rows whose logits go back.
    ///
    /// It counts every value that the driver makes in the step, for a backend whose `range` is a view of its
    /// value (a device): what lives through a visit, the largest part of a visit, and the end. The rings of a step
    /// wait in the states to its end, so one more set of rings a chat is in it. The part over keys is in it with
    /// the byte limit of the layout, or with one row where one row is above that limit.
    /// It does not count what `Weights` holds inside a product, or what a launch holds inside the backend.
    /// The part over keys is counted for the wrapped launch of its arms (one launch for the rows of a group). The
    /// other parts are counted with room: each result once more, as a launch for each row and a join hold it.
    pub fn transient_words(&self,rows:usize,chats:usize,ends:usize,logits:usize)->usize {
        let d=self.run.description;
        let (r,c,w,taps)=(rows as u128,chats.min(rows) as u128,self.run.width() as u128,d.conv.kernel as u128);
        let (messages,rank,head)=(d.core.end.saturating_sub(d.core.start) as u128*MESSAGES as u128,d.core.gate_rank as u128,d.head_rows as u128);
        // The rings of each chat, and the window planes of a chat with many rows (`state::many::narrow`), wait to the end.
        let pending=c*self.layouts.iter().map(|l|2*(l.kv_taps*l.kv_channels+l.taps*l.channels) as u128+if l.wide().rows>l.rows {2*(l.rows*l.row_words) as u128} else {0}).sum::<u128>();
        // One mix of `asked` queries over `vectors` vectors a row: the joined list in both orders, the dots, the
        // result as parts and joined, two tables.
        let blend=|asked:u128,vectors:u128|w*r*(2*vectors+2*asked)+3*asked*r*vectors+r*(vectors+asked);
        // One gate: its two norms, the pairs in both orders, the three products as parts and joined, the result.
        let gate=8*w*r+3*rank*r+4*r;
        // `list` is the length of the block list, `kept` its length at the saved start of the loop.
        let (mut list,mut kept,mut most)=(0u128,0u128,0u128);
        for visit in &d.visits {
            let (Some(spec),Some(layout))=(usize::try_from(visit.layer).ok().and_then(|layer|d.layers.get(layer)),usize::try_from(visit.slot).ok().and_then(|slot|self.layouts.get(slot))) else {return usize::MAX;};
            let Ffn::Experts(experts)=spec.ffn else {return usize::MAX;};
            let (a,first)=(spec.attention,visit.layer==d.core.start);
            // The first loop layer of a later pass: the mix of all messages over the blocks of the pass before, the
            // mix result beside its copy by row, then the gate norms of all messages. The messages of the pass before
            // and their norms are gone by then, and the new ones are counted in `through`.
            let enter=if first && visit.messages {(w*r+blend(messages,list.saturating_sub(kept).max(1))+messages*w*r).saturating_sub(2*messages*w*r)} else {0};
            if first {if visit.messages {list=kept;} else {kept=list;}}
            // The mix for attention, then the three gates beside the three inputs.
            let asked=spec.attention_queries as u128;
            let inputs=(if asked>0 {w*r*asked+blend(asked,list+1)} else {0}).max(if visit.messages {3*w*r+gate} else {0});
            // The attention half: the normed inputs, the query side, the keys and the values with their rings, one
            // group of rows over keys at the largest view (the relative term, the scores and the weights), the
            // output and its rings.
            let (heads,kv,kt)=(a.query_heads as u128,a.kv_heads as u128,layout.kv_taps as u128);
            let (out,moved,keys)=(heads*a.head_dim as u128,heads*a.relative.dim as u128,kv*a.head_dim as u128);
            let one=heads*(2*layout.wide().rows as u128+a.relative.extent as u128);
            // The wide planes of a window layer for each chat with many rows (`state::many::widen`), through the attention.
            let widened=if layout.wide().rows>layout.rows {2*c*(layout.wide().rows*layout.row_words) as u128} else {0};
            let group=(layout.key_bytes as u128/4).max(one).min(r*one);
            let attention=w*r*(taps+6)+2*c*taps*w+5*out*r+3*moved*r+2*heads*r+keys*(r*(kt+6)+2*c*kt)+group+r*(2*kt+2*taps+2*kv*kt+8)+c*(kt+taps)+widened;
            // The expert half: the mix and the gate; the router and the picked experts (the input rows by expert, the
            // two products as parts and joined, the activation, the rows back in pick order); the ring; the last sum.
            if spec.boundary {list+=1;}
            let (count,picks,inner)=(experts.count as u128,experts.selected as u128,experts.width as u128);
            let input=(w*r+blend(1,list+1)).max(if visit.messages {2*w*r+gate} else {0});
            let sum=3*w*r+r*(4*count+7*picks)+r*picks*(w+4*inner).max(3*w+3*inner);
            let ring=w*(2*c*taps+r)+r*(taps+1)+c*taps;
            // What lives through the visit: the block list, the saved start and the last plain norm; the messages and
            // their gate norms;
            // the stream before and after, the three inputs, the residual, the input and the sum of the experts, the
            // ring of each row; the rings that wait.
            let through=w*r*(list+2+if visit.messages {2*messages} else {0}+taps+9)+pending+2*r;
            most=most.max(through+enter.max(inputs).max(attention).max(input).max(sum).max(ring).max(2*w*r));
        }
        // The end: the stream, the block list and the end mix; for the rows that run it the last norm, the head
        // product and the capped logits; the logits that go back.
        let last=if d.visits.last().is_some_and(|visit|visit.messages) {2*messages} else {0};
        let end=w*r*(list+4+last)+pending+blend(1,list+1)+ends as u128*(2*w+2*head+1)+logits as u128*(head+1);
        // The index tables of the swaps of the step (`Run::swap`), kept until its end: one of `k x rows` words for
        // each count `k` of vectors a row that a mix or a pair swaps (2 to the longest list and one more, the three
        // inputs, the messages), and the tables of the dots of a mix (the queries times the list, a row).
        let longest=d.visits.len() as u128+2;
        let tables=r*((2..=longest).sum::<u128>()+3+messages+messages.max(3)*longest);
        usize::try_from(most.max(end)+tables).unwrap_or(usize::MAX)
    }
    /// A bound of the device words that a step of `transient_words` holds above what was alive before it: the words
    /// of its values plus the rest of the last page of each device buffer (1,023 words a buffer at most). Inside a
    /// resident queue a vector is whole pages of 4,096 bytes (`reserve_policy::vector_capacity` of the gpu crate).
    ///
    /// What this does not count, and why that is right: a buffer that a value of the step freed waits for a stream
    /// wait, and a spare of the pool waits for a value of its size. Both are counted by the device
    /// (`peak_device_bytes`, `accounted_bytes`) but neither is needed: where an allocation would pass the budget, the
    /// backend waits for the stream, which frees the waiting buffers into the pool, and the pool gives its spares
    /// back to the device (`Backend::funded_buffer`, `make_room`). So with the budget of the backend set to what the
    /// ledger admits, a step needs no more than this above what was alive before it.
    pub fn device_words(&self,rows:usize,chats:usize,ends:usize,logits:usize)->usize {
        self.transient_words(rows,chats,ends,logits).saturating_add(self.transient_buffers(rows,chats).saturating_mul(1023))
    }
    /// A bound of the count of device buffers of a step that live at one time: the values that live through a
    /// visit (the block list, the messages, the rows of a ring), the per-row values of a product of rows before its
    /// join and of the groups over keys (three a row), the values of each chat (four, and the window planes of a chat with many rows), the values of each expert
    /// that rows picked (four), and 64 more for the rest of a visit. The meter of the device counts the live buffers
    /// of steps of the two made-up checkpoints under it (test of `meter.rs`).
    pub fn transient_buffers(&self,rows:usize,chats:usize)->usize {
        let d=self.run.description;
        let messages=d.core.end.saturating_sub(d.core.start) as usize*MESSAGES;
        let blocks=d.visits.iter().filter(|visit|usize::try_from(visit.layer).ok().and_then(|layer|d.layers.get(layer)).is_some_and(|spec|spec.boundary)).count();
        let experts=d.layers.iter().map(|spec|match spec.ffn {Ffn::Experts(e)=>(e.count as usize).min(rows.saturating_mul(e.selected as usize)),_=>0}).max().unwrap_or(0);
        let taps=d.conv.kernel as usize;
        // A chat with many rows keeps two window planes a window layer to the end of the step, and two wide ones in a visit.
        let windows=self.layouts.iter().filter(|layout|layout.wide().rows>layout.rows).count();
        let planes=if windows>0 {2*windows+2} else {0};
        [blocks,messages,taps,64].iter().sum::<usize>().saturating_add(rows.saturating_mul(3)).saturating_add(chats.min(rows).saturating_mul(4+planes)).saturating_add(experts.saturating_mul(4))
    }
    /// The state of a new chat.
    pub fn clean(&self)->Result<State<C::Value>> {state::clean(self.run.compute,&self.layouts)}
    /// Give every ring of `state` a value of its own. No word of the state changes.
    ///
    /// Why: after a step of many chats the rings of a chat are ranges of values that hold the rings of all chats of
    /// that step (`state::row_rings`). On a backend whose range is a view (a device) such a value lives while one
    /// chat still reads it. The next step of a chat gives it new rings, so where every chat goes on, a value of a
    /// step lives one step more. But a chat that takes no later step keeps the rings of the chats that ended: as
    /// many sets of rings as its last step had chats. Call this for a chat that waits with its state kept, or that
    /// goes on with the token path (`step`, `read`), which writes its rings in place.
    /// Cost: one new value and one copy for each of the four rings of each visit. Not for every step.
    pub fn detach(&self,state:&mut State<C::Value>)->Result<()> {
        let c=self.run.compute;
        need(state.visits.len()==self.layouts.len(),"the state differs from the visits of the description")?;
        for ring in state.visits.iter_mut().flat_map(|visit|visit.rings.iter_mut()) {
            let words=c.words(ring);let mut own=c.zeros(words)?;c.rows_fill(&mut own,0,ring,words)?;*ring=own;
        }
        Ok(())
    }
    /// The input of the first visit: the stored row of the token under the weighted norm, BF16 values
    /// (reference lines 1240-1246). Every stored row can be read, also a row that the head cannot write.
    pub fn embedding(&self,token:u32)->Result<C::Value> {
        let (c,w,d,width)=(self.run.compute,self.run.weights,self.run.description,self.run.width());
        let table=w.words(Weight::Embedding)?;
        need(u64::from(token)<d.vocab && table.len() as u64==d.vocab*width as u64,"the token is outside the stored embedding rows")?;
        let row=c.upload(&table[token as usize*width..(token as usize+1)*width])?;
        rounded_norm(c,&row,1,width,d.epsilon,Some(w.words(Weight::EmbeddingNorm)?))
    }
    /// The end of a token step: the end mix, the last norm, the head on the rows that the model can write, the logit
    /// cap and the top row (reference lines 1442-1443 and 1814-1816).
    /// The head product is 32-bit sums; the cap arm rounds each to BF16 as it reads it. The cap is
    /// `tanh(x * r) * cap` with `r` the `reciprocal` of the cap: the arm gets the cap and `r` and divides nothing.
    pub fn end(&self,token:&Token<C::Value>,stream:&C::Value)->Result<Step<C::Value>> {
        need(self.run.rows(stream)?==1,"the end of a token step takes one row")?;
        let (logits,tops)=self.capped(&token.end_mix(&self.run,stream)?,1)?;
        Ok(Step {logits,top:tops[0]})
    }
    /// The end after the end mix, for `rows` rows of it (`[rows][width]`): the last norm, the head, the cap and the
    /// top row of each. One norm launch, one head product, one cap launch and one read of the top rows for all rows.
    /// It gives the capped logits, `[rows][head rows]`, and the top rows.
    fn capped(&self,mixed:&C::Value,rows:usize)->Result<(C::Value,Vec<u32>)> {
        let (c,w,d,width)=(self.run.compute,self.run.weights,self.run.description,self.run.width());
        let head=state::size(d.head_rows)?;
        let normed=rounded_norm(c,mixed,rows,width,d.epsilon,Some(w.words(Weight::LastNorm)?))?;
        let product=w.product_rows(c,Weight::Head,&normed,rows)?;need(head.checked_mul(rows)==Some(c.words(&product)),"the head product differs from the head rows")?;
        let mut s=shape(Op::Softcap);s.rows=number(rows*head)?;s.bound=d.caps.logits as f32;s.epsilon=reciprocal(d.caps.logits);
        let logits=c.operation(s,[Some(&product),None,None,None],&[])?;
        let tops=match c.rows_greedy(&logits,rows)? {
            Some(picked)=>{need(picked.len()==rows && picked.iter().all(|&top|u64::from(top)<d.head_rows),"the device picked no head row")?;picked},
            None=>{let words=c.read(&logits)?;(0..rows).map(|row|greedy_logits(&words[row*head..(row+1)*head],head)).collect::<Result<Vec<_>>>()?},
        };
        Ok((logits,tops))
    }
    /// One token: its embedding, every visit of the description in its order, and the end. The token is the one at
    /// `state.position`. The position moves on only when the whole step is done.
    /// `seen` gets the place of each visit in run order and the stream after that visit, for checks; it changes no word.
    /// A step that fails leaves the position as it was, and the same token can be run again on the same state
    /// (see `attention::half`).
    /// The caller owns the work scope of the backend (`Compute::work`), as with the other decoders of this crate:
    /// this call opens none.
    pub fn step_seen(&self,state:&mut State<C::Value>,token:u32,seen:impl FnMut(usize,&C::Value)->Result<()>)->Result<Step<C::Value>> {
        let (mix,stream)=self.visits(state,token,seen)?;
        let step=self.end(&mix,&stream)?;
        state.position+=1;Ok(step)
    }
    /// `step_seen` with nothing to see.
    pub fn step(&self,state:&mut State<C::Value>,token:u32)->Result<Step<C::Value>> {self.step_seen(state,token,|_,_|Ok(()))}
    /// One token whose logits nobody reads: a prompt token before the last one. Every visit runs and the end does
    /// not: no end mix, no last norm, no head product, no cap and no top row.
    /// The state after it is the state after `step` of the same token, word for word: the end writes no state.
    /// The rules of `step_seen` on a failed step and on the work scope hold here too.
    pub fn read(&self,state:&mut State<C::Value>,token:u32)->Result<()> {
        self.visits(state,token,|_,_|Ok(()))?;
        state.position+=1;Ok(())
    }
    /// One step of many rows: the next tokens of each chat of `chats`, all in one pass over the visits.
    /// One design for a piece of a prompt (many rows of one chat), for one new token of many chats, and for a mix.
    ///
    /// Exactness: a row has the words of that row in steps of one row (`step`, `read`). The words of a chat do not
    /// depend on the other chats of the step or on how its tokens are cut into steps.
    ///
    /// Refused before any launch and before any value is made, with the number of the chat in the step in the
    /// text: a chat with no token; a state of another driver (another count of visits, or planes and rings of
    /// other sizes, as a driver with another capacity makes them); a state with a ring
    /// that still waits; a chat that would pass the capacity or the model's longest length; a token outside the
    /// stored embedding rows. Refused for the whole step: more rows than `limits().step_rows`; more rows with an end
    /// than one launch of the logits takes.
    ///
    /// The end runs only for the wanted rows (`Wanted`). A row that is not wanted gives nothing back.
    ///
    /// A step that fails moves no position of any chat, and the same step, or the same rows without one chat, can
    /// run again on the same states with the same words (the rule of `step_seen`). The rings of a step wait in the
    /// states until the whole step is done (`state::commit`), and a step that fails drops them (`state::discard`).
    /// The key and value rows that it wrote are rows of positions that no chat has reached: no row reads them
    /// before a later step writes them again.
    ///
    /// `seen` gets the place of each visit in run order and the stream after that visit, `[R][width]` in row order.
    /// The caller owns the work scope of the backend, as for `step_seen`.
    ///
    /// Launches: the mix, the expert half and the end take all rows in their launches (piece D2). The picked
    /// expert numbers are one read a visit, and the top rows are one read a step. The attention half and the rings
    /// run as `attention::half_rows` and `state::row_rings` run them (piece D1).
    pub fn step_rows_seen(&self,chats:&mut [ChatStep<'_,C::Value>],seen:impl FnMut(usize,&C::Value)->Result<()>)->Result<StepRows<C::Value>> {
        let rows=self.checked(chats)?;
        self.run.clear_tables();
        let result=self.rows_pass(chats,&rows,seen);self.run.clear_tables();
        let mut states=chats.iter_mut().map(|chat|&mut *chat.state).collect::<Vec<_>>();
        match result {
            Ok(step)=>{state::commit(&mut states,&rows)?;Ok(step)},
            Err(error)=>{state::discard(&mut states);Err(error)},
        }
    }
    /// `step_rows_seen` with nothing to see.
    pub fn step_rows(&self,chats:&mut [ChatStep<'_,C::Value>])->Result<StepRows<C::Value>> {self.step_rows_seen(chats,|_,_|Ok(()))}
    /// Every refusal of a step of rows, before any launch. It gives the row list.
    fn checked(&self,chats:&[ChatStep<'_,C::Value>])->Result<Rows> {
        let (c,d)=(self.run.compute,self.run.description);
        need(!chats.is_empty() && d.visits.len()==self.layouts.len(),"a step of rows has no chat")?;
        for (number,chat) in chats.iter().enumerate() {
            let refuse=|ok:bool,why:&str|if ok {Ok(())} else {Err(format!("loop decoder: chat {number} of the step: {why}"))};
            refuse(chat.state.visits.len()==self.layouts.len(),"the state differs from the visits of the description")?;
            // A state of a driver with another capacity has planes of another size (the step row limit changes none).
            refuse(chat.state.visits.iter().zip(&self.layouts).all(|(visit,layout)|{
                let plane=layout.rows*layout.row_words;
                c.words(&visit.keys)==plane && c.words(&visit.values)==plane && Ring::ALL.iter().all(|&ring|{let (channels,taps)=layout.ring_size(ring);c.words(&visit.rings[ring as usize])==channels*taps})
            }),"the state has the sizes of another driver (another capacity)")?;
            refuse(chat.state.visits.iter().all(|visit|visit.next.iter().all(Option::is_none) && visit.planes.is_none()),"a ring of the state still waits for the end of a step")?;
            refuse(!chat.tokens.is_empty(),"it has no token")?;
            let end=chat.state.position.checked_add(chat.tokens.len() as u64);
            refuse(end.is_some_and(|end|end<=self.capacity),"the chat is longer than the state's capacity")?;
            refuse(end.is_some_and(|end|end<=d.max_positions),"the chat is at the model's longest length")?;
            refuse(chat.tokens.iter().all(|&token|u64::from(token)<d.vocab),"a token is outside the stored embedding rows")?;
        }
        let rows=Rows::new(chats.iter().map(|chat|ChatRows {first:chat.state.position,count:chat.tokens.len()}).collect())?;
        need(rows.total()<=self.limits.step_rows,"the step has more rows than the step row limit of this driver")?;
        // The cap of the logits is one launch for all rows with an end: its count of words is a 32-bit number.
        let ends=chats.iter().map(|chat|match chat.wanted {Wanted::None=>0,Wanted::Last=>1,Wanted::All=>chat.tokens.len()}).sum::<usize>();
        need(ends.checked_mul(state::size(d.head_rows)?).is_some_and(|words|u32::try_from(words).is_ok()),"the step has more rows with an end than one launch of the logits takes")?;
        Ok(rows)
    }
    /// The pass of a step of rows over the visits, and the end of the wanted rows. No position moves here.
    fn rows_pass(&self,chats:&mut [ChatStep<'_,C::Value>],rows:&Rows,mut seen:impl FnMut(usize,&C::Value)->Result<()>)->Result<StepRows<C::Value>> {
        let (run,c,w,d,width,total)=(&self.run,self.run.compute,self.run.weights,self.run.description,self.run.width(),rows.total());
        // The stored rows of all tokens of the step in one value, then one norm launch for all rows.
        let table=w.words(Weight::Embedding)?;need(table.len() as u64==d.vocab*width as u64,"the stored embedding rows differ from the description")?;
        let stored=chats.iter().flat_map(|chat|chat.tokens.iter()).flat_map(|&token|table[token as usize*width..(token as usize+1)*width].iter().copied()).collect::<Vec<_>>();
        let mut stream=rounded_norm(c,&c.upload(&stored)?,total,width,d.epsilon,Some(w.words(Weight::EmbeddingNorm)?))?;
        let mut mix=Token::new();
        for (place,visit) in d.visits.iter().enumerate() {
            let slot=state::size(visit.slot)?;let layout=&self.layouts[slot];
            let mut kept=chats.iter_mut().map(|chat|&mut chat.state.visits[slot]).collect::<Vec<_>>();
            let keep=mix.enter(run,visit,stream)?;
            let inputs=mix.attention_inputs(run,visit,&keep)?;
            let residual=attention::half_rows(run,visit,layout,rows,&mut kept,&keep,&inputs)?;
            let input=mix.expert_input(run,visit,&keep,&residual)?;
            let sum=experts::sum(run,visit,&input)?;
            let (rings,words,window)=state::conv_rings(c,layout,Ring::AfterExperts,rows,&mut kept,&sum)?;
            stream=experts::close(run,visit,&rings,&words,window,&residual)?;
            seen(place,&stream)?;
        }
        // The end, for the wanted rows only: the end mix of those rows, then one norm, one head product, one cap.
        let (mut wanted,mut asked)=(Vec::new(),Vec::new());
        for (number,chat) in chats.iter().enumerate() {
            let (first,count)=(rows.offset(number),chat.tokens.len());
            for index in match chat.wanted {Wanted::None=>first..first,Wanted::Last=>first+count-1..first+count,Wanted::All=>first..first+count} {
                if chat.logits {asked.push(wanted.len());}
                wanted.push(index);
            }
        }
        if wanted.is_empty() {return Ok(StepRows {tops:Vec::new(),logits:None});}
        let (logits,tops)=self.capped(&mix.end_mix_rows(run,&stream,&wanted)?,wanted.len())?;
        Ok(StepRows {tops,logits:if asked.is_empty() {None} else {Some(gather(c,&logits,&asked,state::size(d.head_rows)?)?)}})
    }
    /// The embedding of the token at `state.position` and every visit of the description in its order. It gives the
    /// mix state of the token and the stream after the last visit. The position does not move here.
    fn visits(&self,state:&mut State<C::Value>,token:u32,mut seen:impl FnMut(usize,&C::Value)->Result<()>)->Result<(Token<C::Value>,C::Value)> {
        let (run,c,d)=(&self.run,self.run.compute,self.run.description);
        need(state.visits.len()==self.layouts.len() && d.visits.len()==self.layouts.len(),"the state differs from the visits of the description")?;
        need(state.position<d.max_positions,"the chat is at the model's longest length")?;
        let position=state.position;
        let (mut mix,mut stream)=(Token::new(),self.embedding(token)?);
        for (place,visit) in d.visits.iter().enumerate() {
            let slot=state::size(visit.slot)?;let (layout,kept)=(&self.layouts[slot],&mut state.visits[slot]);
            let keep=mix.enter(run,visit,stream)?;
            let inputs=mix.attention_inputs(run,visit,&keep)?;
            let residual=attention::half(run,visit,layout,kept,position,&keep,&inputs)?;
            let input=mix.expert_input(run,visit,&keep,&residual)?;
            let sum=experts::sum(run,visit,&input)?;
            let word=state::write_ring(c,layout,Ring::AfterExperts,&mut kept.rings,position,&sum)?;
            stream=experts::close(run,visit,&kept.rings[Ring::AfterExperts as usize],&word,0,&residual)?;
            seen(place,&stream)?;
        }
        Ok((mix,stream))
    }
}
#[cfg(test)]
mod tests;
#[cfg(test)]
mod meter;
