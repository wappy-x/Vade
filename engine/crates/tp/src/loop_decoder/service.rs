//! The text service of the looped block family on one backend: the engine and the session that the HTTP scheduler
//! drives, over the driver of this folder. Many rows a step: a prompt is read in pieces of rows, and the new tokens
//! of several chats run in one step.
//!
//! `Service` is the `spark_serve::Engine` of one checkpoint on one backend. `Chat` is its `spark_serve::Session`: it
//! owns one `State` of the driver. Every token goes through `Driver::step_rows`. The service holds no transport: the
//! driver runs on one device.
//!
//! How the tokens become steps:
//! - A prompt: the scheduler gives one chat a piece of at most the step row limit of the driver
//!   (`Driver::limits`), and the piece is one step. Only the last row of the last piece runs the end of the
//!   decoder. No other prompt row runs the head.
//! - New tokens: with two or more chats the scheduler gives one row for each chat in one call (`step_batch`), and
//!   the service packs these rows into one step. A step never takes more rows than the limit: rows that do not
//!   fit go on in the next step of the same call.
//! - The words of a chat do not depend on the other chats of a step or on how its tokens are cut into steps
//!   (the exactness rule of `rows`).
//!
//! What a step answers:
//! - A row that the scheduler marks greedy gets the driver's own top row: the largest capped logit, and the lower
//!   row when two logits are equal. No logit leaves the backend for it.
//! - Any other row gets the capped logits of the rows that the head can write, read from the backend.
//!
//! The vocabulary of the service is the count of rows that the head can write (`LoopTextDescription::head_rows`). A
//! stored embedding row above it is no token of the tokenizer, so a prompt that names one is refused by the scheduler.
//!
//! Limits, in two parts. Before any load `admit` reads the byte ledger of the load plan
//! (`engine_model::plan::loop_state`) for steps of the step row limit: the weights and one chat must fit. The
//! state of a chat is allocated whole when the chat starts, so every request reserves the bytes of one whole chat.
//! When the driver exists, `Service::new` takes the driver's own bound of the values of one step
//! (`Driver::device_words`) off the room once, because one step runs at a time. The chats that fit in the rest
//! are the chats that the service runs at one time.
//!
//! The stop set is the list that the caller gives. The service start gives the EOS list of the checkpoint and nothing
//! else. The end of one message is not in it: the model goes on with its next message.
//!
//! Every size comes from the description of the checkpoint. No line here names a model or a count of ranks.
use super::{ChatStep,Driver,Wanted,mix::Weights,rows::Limits,state::State};
use crate::{Result,bounded_decoder::Compute};
use engine_model::plan::{description::LoopTextDescription,loop_load::LoadBytes,loop_state::{LoopLedger,LoopStatePlan,context_limit}};
use spark_serve::{BatchError,BatchOutput,BatchRows,Predictions,cache::CacheLimits,protocol::{DraftMode,Request},scheduler::Config};
use std::{cell::{Cell,RefCell},time::{Duration,Instant}};
pub mod start;
#[cfg(test)] mod tests;

/// The limits of a step of this service: the most rows of one step, and the most bytes of the values over the
/// keys of one launch group. They are sizes of the service, not of a model. The step row limit is also the
/// prompt piece of the scheduler, and it gives the extra ring rows of a chat (`rows::Limits`).
pub const STEP_LIMITS:Limits=Limits {step_rows:1024,key_bytes:1<<30};
const DRAFT_REFUSAL:&str="this text service decodes by plain steps: drafting is not built for this model family; leave draft and mtp_depth out, or send draft plain and mtp_depth 0";

fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(format!("loop text service: {why}"))}}
fn size(n:u64)->Result<usize> {usize::try_from(n).map_err(|_|"loop text service: a size exceeds this host".to_owned())}

/// What the memory rule admits for one start, before any load. Every number comes from the byte ledger of the load plan.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Admitted {
    /// Tokens of one chat, prompt and answer together.
    pub context:usize,
    /// The ledger: the weights and one chat, in bytes. The state of the chat has the extra ring rows of the step
    /// row limit.
    pub ledger:LoopLedger,
    /// The budget of device memory that the ledger was admitted against.
    pub budget:u64,
    /// The limits of a step that the ledger was read for. The service builds its driver with them.
    pub limits:Limits,
    /// The largest context whose ledger fits the budget.
    pub largest_context:u64,
}
impl Admitted {
    /// Bytes of one chat in the ledger: its state and the values of one row of a step.
    pub fn chat_bytes(&self)->Result<usize> {size(self.ledger.chat())}
    /// The budget less the weights: the room for the values of a step and for the chats.
    pub fn room(&self)->Result<usize> {size(self.budget-self.ledger.weights)}
    /// What the service gets: the limits, the bytes of one chat, the room, and no cap on the chats but the step
    /// row limit.
    pub fn seats(&self)->Result<Seats> {Ok(Seats {limits:self.limits,chat:self.chat_bytes()?,room:self.room()?,most:self.limits.step_rows})}
}
/// Admit `context` tokens of one chat under `budget` bytes of device memory, before any device work, for steps of
/// `limits`. `bucket` is the row rule of the device for a launch that grows with the context
/// (`causal_ops::BUCKET_ROWS`). A context that the model does not take, and a context whose ledger does not fit,
/// are refused with the limit in the text. The values of a step are not in this ledger: the driver bounds them
/// (`Service::new`).
pub fn admit(description:&LoopTextDescription,load:&LoadBytes,context:usize,budget:u64,bucket:u64,limits:Limits)->Result<Admitted> {
    let state=LoopStatePlan::for_steps(description,context as u64,bucket,limits.step_rows as u64).map_err(|e|format!("text service context refusal: {e}"))?;
    let ledger=LoopLedger::new(load,&state).map_err(|e|e.to_string())?;
    // `context_limit` plans one row a step. The extra ring rows do not change with the context: they come off the
    // budget that it sees.
    let extra=state.fixed_bytes().and_then(|rows|Ok(rows-LoopStatePlan::new(description,context as u64,bucket)?.fixed_bytes()?)).map_err(|e|e.to_string())?;
    let largest_context=context_limit(description,load,bucket,budget.saturating_sub(extra)).map_err(|e|e.to_string())?;
    ledger.admit(budget,largest_context).map_err(|refusal|format!("text service memory refusal: {refusal}"))?;
    Ok(Admitted {context,ledger,budget,limits,largest_context})
}
/// What a service is built with besides its checkpoint: the limits of a step, the bytes of one chat in the ledger
/// (`chat`), the bytes for the values of a step and for all chats (`room`: the budget less the weights), and the
/// most chats that the caller wants at one time (`most`).
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Seats {pub limits:Limits,pub chat:usize,pub room:usize,pub most:usize}

/// The engine of the text service: the driver of one checkpoint on one backend, and the limits of its chats.
pub struct Service<'a,C:Compute,W:Weights<C>> {
    driver:Driver<'a,C,W>,compute:&'a C,rows:usize,context:usize,eos:Vec<u32>,
    /// Bytes that one chat reserves, the bytes of the values of one step, what the chats can use, and the chats.
    chat_bytes:usize,step:usize,pool:usize,chats:usize,
    /// Chats that are open now, and the number of the next chat.
    live:Cell<usize>,next:Cell<u64>,
    /// The state of each open chat, by its slot. The service holds the states, so a call can reach a chat that
    /// waits (`settle`). A free slot is `None`.
    slots:RefCell<Vec<Option<Slot<C::Value>>>>,
    /// Waiting chats that were given rings of their own (`Driver::detach`), over the life of the service.
    detached:Cell<u64>,
    /// Steps that ran the token pass (`run`, one new token of one chat alone), since the start.
    token_steps:Cell<u64>,
}
/// The state of one open chat, and whether its rings are views of the values of a step of several chats.
struct Slot<V> {state:State<V>,shared:bool}
impl<'a,C:Compute,W:Weights<C>> Service<'a,C,W> {
    /// `context` is the longest chat in tokens. `eos` is the stop set: the caller gives the EOS list of the checkpoint.
    /// `seats` gives the limits of a step and the bytes (`Admitted::seats`).
    ///
    /// The chats of the service are counted here, with the driver's own numbers:
    /// - The values of the largest step come off the room once (`Driver::device_words` for a step of the row limit
    ///   over all chats of the service, where every row runs the end and gives its logits back: the words of its
    ///   values and the rest of the last page of each device buffer).
    /// - A chat reserves its bytes of the ledger and, for each other chat of the service, one more set of
    ///   convolution rings: the rings of a chat are views of one block of its last step, and a chat that waits
    ///   keeps that block alive.
    /// - The count is the most chats that fit so, at most `seats.most` and at most the step row limit (a step of
    ///   new tokens has one row a chat). No chat fits: the start is refused with the numbers.
    pub fn new(compute:&'a C,weights:&'a W,description:&'a LoopTextDescription,context:usize,eos:Vec<u32>,seats:Seats)->Result<Self> {
        need(context>0 && context as u64<=description.max_positions,"the context is zero or above the model's longest chat")?;
        let rows=size(description.head_rows)?;
        need(!eos.is_empty(),"the stop set is empty: a chat could not end")?;
        need(eos.iter().all(|&id|(id as usize)<rows),"a stop token is outside the rows that the head can write")?;
        need(eos.iter().enumerate().all(|(place,id)|!eos[..place].contains(id)),"the stop set names one token twice")?;
        let limit=seats.limits.step_rows;
        need(seats.most>0 && seats.most<=limit,"the count of chats is zero or above the step row limit")?;
        let driver=Driver::with_limits(compute,weights,description,context as u64,seats.limits)?;
        need(driver.state_words().checked_mul(4).is_some_and(|bytes|bytes<=seats.chat),"the state of one chat is larger than the bytes that the ledger reserves for a chat")?;
        let rings=driver.layouts().iter().map(|layout|2*(layout.kv_taps*layout.kv_channels+layout.taps*layout.channels)).sum::<usize>().saturating_mul(4);
        let hold=|chats:usize|seats.chat.saturating_add((chats-1).saturating_mul(rings));
        let step=|chats:usize|driver.device_words(limit,chats,limit,limit).saturating_mul(4);
        let fits=|chats:usize|chats.saturating_mul(hold(chats)).saturating_add(step(chats))<=seats.room;
        let chats=(1..=seats.most).take_while(|&chats|fits(chats)).last().unwrap_or(0);
        if chats==0 {return Err(format!("text service memory refusal: CONTEXT={context}; room={}; step_values={}; session_pool={}; full_session={}; no chat fits beside the values of one step of {limit} rows: reduce CONTEXT",seats.room,step(1),seats.room.saturating_sub(step(1)),seats.chat));}
        let (step,chat_bytes)=(step(chats),hold(chats));
        Ok(Self {driver,compute,rows,context,eos,chat_bytes,step,pool:seats.room-step,chats,live:Cell::new(0),next:Cell::new(0),slots:RefCell::new(Vec::new()),detached:Cell::new(0),token_steps:Cell::new(0)})
    }
    /// Waiting chats that were given rings of their own, over the life of the service.
    pub fn detached(&self)->u64 {self.detached.get()}
    /// How many steps ran the token pass of the driver (one new token of one chat alone), since the start.
    pub fn token_steps(&self)->u64 {self.token_steps.get()}
    /// The end of a call. After a step of several chats the rings of each of them are views of one value of that
    /// step (`state::row_rings`). A chat that takes its next step gets new rings, so that value lives one step
    /// more. A chat that waits would keep it alive for the chats that went on or ended. So every chat whose rings
    /// are shared and that was not in this call (`called`, by slot) gets rings of its own now (`Driver::detach`),
    /// once: after that it holds only its own state while it waits. A chat that goes on needs no copy.
    fn settle(&self,slots:&mut [Option<Slot<C::Value>>],called:&[usize])->Result<()> {
        for (place,slot) in slots.iter_mut().enumerate() {
            let Some(slot)=slot.as_mut().filter(|slot|slot.shared && !called.contains(&place)) else {continue};
            self.compute.work(||self.driver.detach(&mut slot.state))?;
            slot.shared=false;self.detached.set(self.detached.get()+1);
        }
        Ok(())
    }
    /// Chats that are open now.
    pub fn live(&self)->usize {self.live.get()}
    /// The most chats at one time.
    pub fn chats(&self)->usize {self.chats}
    /// The limits of a step of the driver of this service.
    pub fn limits(&self)->Limits {self.driver.limits()}
    /// Bytes of the state that the driver allocates for one chat.
    pub fn state_bytes(&self)->usize {self.driver.state_words()*4}
    /// Bytes that one chat reserves: its bytes of the ledger and the rings that it can keep alive for the other chats.
    pub fn chat_bytes(&self)->usize {self.chat_bytes}
    /// The driver's bound of the bytes of the values of one step. They are taken off the room once.
    pub fn step_bytes(&self)->usize {self.step}
    /// What the chats can use: the room less the values of one step.
    pub fn pool(&self)->usize {self.pool}
    /// The scheduler settings of this service: its chats, a prompt piece of the step row limit, plain decoding,
    /// and no kept prompt.
    pub fn schedule(&self)->(CacheLimits,Config) {
        let pool=self.pool;
        (CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:self.context,logits_per_entry:self.rows,idle_ttl:Duration::from_secs(300)},
        Config {active:self.chats,connections:self.chats.max(32),prefill_chunk:self.driver.limits().step_rows,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,
            draft:DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()})
    }
    /// One driver step in one work scope of the backend: the step, and the read of the logits that its chats
    /// asked for. The answer is the top rows of the wanted rows and the logits that were read, both in row order.
    /// A step that fails moves no position (`Driver::step_rows_seen`).
    /// `alone`: the chats are one chat whose rings are its own (no other chat shares them).
    fn run(&self,chats:&mut [ChatStep<'_,C::Value>],alone:bool)->Result<(Vec<u32>,Vec<f32>)> {
        self.compute.work(|| {
            // One new token of one chat whose rings are its own (the scheduler asks a decode row for `All`, one row: the
            // same as `Last`): the token pass of the driver (`Driver::step`), which
            // runs fewer launches than a pass of rows for one row. The words are the same (the exactness rule of
            // `step_rows`). Rings that another chat shares stay on the pass of rows, which gives the chat rings of its own.
            if alone && let [chat]=chats && let &[token]=chat.tokens && chat.wanted!=Wanted::None {
                let step=self.driver.step(chat.state,token)?;
                self.token_steps.set(self.token_steps.get()+1);
                return Ok((vec![step.top],if chat.logits {self.compute.read(&step.logits)?} else {Vec::new()}));
            }
            let step=self.driver.step_rows(chats)?;
            let logits=match &step.logits {Some(logits)=>self.compute.read(logits)?,None=>Vec::new()};
            Ok((step.tops,logits))
        })
    }
    /// The rows of one scheduler call in driver steps. The rows are packed in their order: a step takes rows until
    /// the step row limit is full, and a row with more tokens than the room goes on in the next step. A row of
    /// one token for each of several chats is so one step.
    fn batch(&self,rows:&mut [BatchRows<'_,Chat<'_,'a,C,W>>])->Result<Vec<Predictions>> {
        let limit=self.driver.limits().step_rows;
        let (mut steps,mut room)=(vec![Vec::new()],limit);
        for (row,batch) in rows.iter().enumerate() {
            let tokens=batch.tokens.len();
            need(tokens>0 || matches!(batch.output,BatchOutput::None),"a row with no token asks for an answer")?;
            let mut from=0;
            while from<tokens {
                if room==0 {steps.push(Vec::new());room=limit;}
                let to=tokens.min(from+room);
                let wanted=match batch.output {BatchOutput::All=>Wanted::All,BatchOutput::Last if to==tokens=>Wanted::Last,_=>Wanted::None};
                steps.last_mut().expect("the list starts with one step").push(Piece {row,from,to,wanted});
                room-=to-from;from=to;
            }
        }
        let mut answers:Vec<(Vec<u32>,Vec<Vec<f32>>)>=rows.iter().map(|_|Default::default()).collect();
        let called:Vec<usize>=rows.iter().map(|batch|batch.session.slot).collect();
        let mut slots=self.slots.borrow_mut();
        for pieces in steps.iter().filter(|pieces|!pieces.is_empty()) {
            let started=Instant::now();
            // A row is in a step at most one time, and two rows are two chats: each slot is taken once.
            let mut open:Vec<Option<&mut Slot<C::Value>>>=slots.iter_mut().map(Option::as_mut).collect();
            let mut taken=Vec::with_capacity(pieces.len());
            for piece in pieces {taken.push(open.get_mut(called[piece.row]).and_then(Option::take).ok_or("loop text service: a row names a chat that is not open")?);}
            let alone=pieces.len()==1 && taken.first().is_some_and(|slot|!slot.shared);
            let mut chats:Vec<ChatStep<'_,C::Value>>=taken.iter_mut().zip(pieces).map(|(slot,piece)|{let batch=&rows[piece.row];
                ChatStep {state:&mut slot.state,tokens:&batch.tokens[piece.from..piece.to],wanted:piece.wanted,logits:!batch.greedy && piece.wanted!=Wanted::None}}).collect();
            let (tops,logits)=self.run(&mut chats,alone)?;
            drop(chats);
            for slot in taken {slot.shared=pieces.len()>1;}
            let (elapsed,mut top,mut word)=(started.elapsed(),0,0);
            for piece in pieces {
                let (batch,tokens)=(&mut rows[piece.row],piece.to-piece.from);
                let count=match piece.wanted {Wanted::All=>tokens,Wanted::Last=>1,Wanted::None=>0};
                batch.session.steps+=tokens as u64;batch.session.stepped+=elapsed;
                let ids=tops.get(top..top+count).ok_or("loop text service: a step gave fewer top rows than its rows asked for")?;
                top+=count;
                if batch.greedy {answers[piece.row].0.extend_from_slice(ids);continue;}
                for _ in 0..count {
                    answers[piece.row].1.push(logits.get(word..word+self.rows).ok_or("loop text service: a step gave fewer logits than its rows asked for")?.to_vec());
                    word+=self.rows;
                }
            }
            need(top==tops.len() && word==logits.len(),"a step gave more answers than its rows asked for")?;
        }
        self.settle(&mut slots,&called)?;
        // A row that wants no answer is answered with no logits, also when it is greedy.
        Ok(rows.iter().zip(answers).map(|(batch,(ids,logits))|if batch.greedy && !matches!(batch.output,BatchOutput::None) {Predictions::Greedy(ids)} else {Predictions::Logits(logits)}).collect())
    }
}
/// A part of one scheduler row in one driver step: the row, its tokens `from..to`, and the rows that run the end.
struct Piece {row:usize,from:usize,to:usize,wanted:Wanted}
/// The request limits of this service at `context` tokens with an answer of at most `output` tokens. They are the
/// limits of the text service of the other family, whole (`row_split::limits`). So a request that names no sampling
/// field is decoded greedy, and a named field sits on a greedy base. The sampling values of the checkpoint are not read.
pub fn limits(context:usize,output:usize)->Result<spark_serve::Limits> {crate::row_split::limits(context,output)}
/// One chat of the service: its state on the backend. It ends when it is dropped.
/// Its state is in the slot `slot` of the service.
pub struct Chat<'s,'a,C:Compute,W:Weights<C>> {service:&'s Service<'a,C,W>,slot:usize,id:u64,steps:u64,stepped:Duration}
impl<C:Compute,W:Weights<C>> Chat<'_,'_,C,W> {
    /// The next tokens of this chat alone, in steps of at most the step row limit. `last` asks for the logits
    /// after the last token: only that row runs the end of the decoder.
    fn read(&mut self,tokens:&[u32],last:bool)->Result<Option<Vec<f32>>> {
        let (service,limit)=(self.service,self.service.driver.limits().step_rows);
        let (pieces,mut answer)=(tokens.len().div_ceil(limit),None);
        let mut slots=service.slots.borrow_mut();
        for (place,piece) in tokens.chunks(limit).enumerate() {
            let (started,wanted)=(Instant::now(),if last && place+1==pieces {Wanted::Last} else {Wanted::None});
            let slot=slots.get_mut(self.slot).and_then(Option::as_mut).ok_or("loop text service: the chat is not open")?;
            let alone=!slot.shared;
            let (_,logits)=service.run(&mut [ChatStep {state:&mut slot.state,tokens:piece,wanted,logits:wanted==Wanted::Last}],alone)?;
            slot.shared=false;
            self.steps+=piece.len() as u64;self.stepped+=started.elapsed();
            if wanted==Wanted::Last {need(logits.len()==service.rows,"the logits differ from the rows that the head can write")?;answer=Some(logits);}
        }
        // The chats in decode wait while this chat reads alone.
        service.settle(&mut slots,&[self.slot])?;
        Ok(answer)
    }
}
impl<C:Compute,W:Weights<C>> spark_serve::Session for Chat<'_,'_,C,W> {
    /// A top row that is no token of the tokenizer fails the request. No other token is picked in its place.
    fn reject_unmapped_greedy(&self)->bool {true}
    fn step(&mut self,token:u32)->Result<Vec<f32>> {self.read(&[token],true)?.ok_or_else(||"loop text service: a step read no logits".to_owned())}
    /// A prompt token before the last one: the end of the decoder does not run for it.
    fn advance(&mut self,token:u32)->Result<()> {self.read(&[token],false).map(|_|())}
    /// A piece of a prompt with the logits after its last token.
    fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>> {self.read(tokens,true)?.ok_or_else(||"loop text service: empty prefill".to_owned())}
    /// A piece of a prompt that is not its last piece: no row runs the end of the decoder.
    fn prefill_advance(&mut self,tokens:&[u32])->Result<()> {self.read(tokens,false).map(|_|())}
    /// The prompt piece is the step row limit: a piece is one step.
    fn prefill_chunk_size(&self,_:usize)->usize {self.service.driver.limits().step_rows}
    /// The state of a chat cannot be copied or wound back: a kept prompt and a fork are not built.
    fn can_fork(&self)->bool {false}
}
impl<C:Compute,W:Weights<C>> Drop for Chat<'_,'_,C,W> {
    fn drop(&mut self) {
        let seconds=self.stepped.as_secs_f64();
        let position=self.service.slots.borrow_mut().get_mut(self.slot).and_then(Option::take).map_or(0,|slot|slot.state.position);
        eprintln!("SERVED CHAT END session={} tokens={} position={position} step_seconds={seconds:.6} tokens_per_second={:.3}",self.id,self.steps,if seconds>0.0 {self.steps as f64/seconds} else {0.0});
        self.service.live.set(self.service.live.get().saturating_sub(1));
    }
}
impl<'a,C:Compute,W:Weights<C>> spark_serve::Engine for Service<'a,C,W> {
    type Session<'s>=Chat<'s,'a,C,W> where Self:'s;
    fn session(&self)->Result<Self::Session<'_>> {
        if let Some(error)=self.compute.fatal_error() {return Err(error);}
        need(self.live.get()<self.chats,"no free chat: every chat of this service is in use")?;
        let started=Instant::now();let state=self.driver.clean()?;
        let id=self.next.get();self.next.set(id.checked_add(1).ok_or("loop text service: chat numbers are used up")?);self.live.set(self.live.get()+1);
        eprintln!("SERVED CHAT session={id} live={} setup_seconds={:.9}",self.live.get(),started.elapsed().as_secs_f64());
        let mut slots=self.slots.borrow_mut();
        let slot=match slots.iter().position(Option::is_none) {Some(free)=>free,None=>{slots.push(None);slots.len()-1}};
        slots[slot]=Some(Slot {state,shared:false});
        Ok(Chat {service:self,slot,id,steps:0,stepped:Duration::ZERO})
    }
    /// A request that asks for drafting is refused before a chat opens, with the reason in the answer.
    fn validate_request(&self,_prompt:&[u32],request:&Request)->Result<()> {
        if request.mtp_depth.is_some_and(|depth|depth>0) || request.draft.is_some_and(|mode|!matches!(mode,DraftMode::Auto|DraftMode::Plain)) {return Err(DRAFT_REFUSAL.into());}
        Ok(())
    }
    fn fatal_error(&self)->Option<String> {self.compute.fatal_error()}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {self.compute.memory_bytes()}
    fn context_limit(&self)->usize {self.context}
    fn vocab_size(&self)->usize {self.rows}
    fn eos_token_ids(&self)->&[u32] {&self.eos}
    /// The whole state of a chat is allocated when the chat starts, at any prompt length.
    fn session_reservation_bytes(&self)->Option<usize> {Some(self.chat_bytes)}
    fn request_reservation_bytes(&self,tokens:usize)->Option<usize> {(tokens>0 && tokens<=self.context).then_some(self.chat_bytes)}
    fn reserve_full_output(&self)->bool {true}
    /// Every chat is the same work for this service: no slot is kept free for one kind of request.
    fn reserve_foreground_slot(&self)->bool {false}
    /// With two or more chats a tick gives one row of new tokens for each chat, and `step_batch` runs them in one step.
    fn shared_decode_rows(&self)->usize {self.chats}
    fn step_batch<'s>(&'s self,rows:&mut [BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<Predictions>,BatchError> {
        self.batch(rows).map_err(|message|BatchError {request:None,message})
    }
}
