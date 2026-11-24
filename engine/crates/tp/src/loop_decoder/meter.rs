//! Checks of a step of many rows on the two made-up checkpoints:
//! - Gate G4 of the rows work: the byte limit of the part over keys and the extra ring rows change no word.
//! - The rule of a step that fails: it runs again on the same states with the same words.
//! - The words that a step holds at one time stay under `Driver::transient_words`.
//! - A step of many rows runs each block arm for all rows in one launch of the rows entry of the arms.
//! - The device buffers that a step needs stay under `Driver::device_words` (`Meter::device`).
//!
//! `Meter` is the host backend with counted values. Every launch runs on `Host`, so the words are those of the
//! host. A value owns a block of words, and a range of a value is a view of that block, as on a device: it keeps
//! the block alive and owns no words. A fill writes in place. So the count is the count of a device that holds one
//! buffer a value. It is no count of what a device holds inside a launch. `Meter` also counts the launches of each
//! operation: its unit launches, and its launches of the rows entry of the arms (`Compute::block_rows`). It refuses
//! a block arm in the general prompt entry (`prompt_ops::Op::Wrapped`), as the device does (Ruling R6g).
use super::{*,attention::hand::bits,fixture::Fixture};
use crate::{bounded_decoder::{ChatGather,Host},mxfp8_projection::Mxfp8Projection};
use engine_model::plan::admission::CheckpointAdmission;
use spark_gpu_probe::{causal_ops::Shape,norm_boundary::NormShape,prompt_ops};
use std::{cell::{Cell,RefCell},collections::BTreeMap,rc::Rc};

/// The words that live now, and the most that lived at one time since `Meter::restart`.
/// With `device` set, the meter also keeps the buffers of the device allocator (`Meter::device`): `held` counts
/// the words of every buffer that the device holds (live, freed and waiting for a drain, and spare), `waiting`
/// the capacities freed since the last drain, `spares` the free buffers by capacity.
#[derive(Default)]
struct Counts {live:Cell<usize>,most:Cell<usize>,device:Cell<bool>,held:Cell<usize>,most_held:Cell<usize>,
    waiting:RefCell<Vec<usize>>,spares:RefCell<BTreeMap<usize,usize>>,buffers:Cell<usize>,
    /// The device buffers of the values that are alive: their words (whole pages) and their count, and both at the
    /// time of the most words.
    pages:Cell<(usize,usize)>,most_pages:Cell<(usize,usize)>,
    /// The budget of the device, if one is set (`Meter::budget`), and whether an allocation passed it.
    budget:Cell<Option<usize>>,refused:Cell<bool>}
/// The words that one value owns, and the capacity of its device buffer. They are counted while the block lives.
struct Block {words:RefCell<Vec<f32>>,counts:Rc<Counts>,capacity:usize}
impl Drop for Block {fn drop(&mut self) {
    self.counts.live.set(self.counts.live.get()-self.words.borrow().len());
    if self.counts.device.get() {self.counts.waiting.borrow_mut().push(self.capacity);let (w,b)=self.counts.pages.get();self.counts.pages.set((w-self.capacity,b-1));}
}}
/// The words of the device buffer of a value of `words` words: whole pages of 4,096 bytes, one page at least
/// (`reserve_policy::vector_capacity` of the gpu crate inside a resident queue; `Backend::vector` asks for one word
/// at least).
pub(super) fn device_capacity(words:usize)->usize {words.max(1).div_ceil(1024)*1024}
/// A value of `Meter`: a range of a block.
pub(super) struct Held {block:Rc<Block>,start:usize,len:usize}
impl std::fmt::Debug for Held {fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {write!(f,"Held({} words)",self.len)}}
/// `launches` counts each launch by (rows entry, operation).
#[derive(Default)]
pub(super) struct Meter {counts:Rc<Counts>,launches:RefCell<BTreeMap<(bool,u32),usize>>}
const NONE:&str="the meter backend has no such launch";
impl Meter {
    fn make(&self,words:Vec<f32>)->Held {
        let (counts,len)=(&self.counts,words.len());counts.live.set(counts.live.get()+len);counts.most.set(counts.most.get().max(counts.live.get()));
        let capacity=device_capacity(len);
        if counts.device.get() {
            // A spare of the same capacity, else a new buffer (`Buffer::pooled` of the gpu crate).
            if !self.take_spare(capacity) {
                if let Some(budget)=counts.budget.get() {
                    // `Backend::funded_buffer`: the pool gives spares back to the device while the new buffer passes the
                    // budget (`make_room`); if it still does and buffers wait, one stream wait frees them into the pool
                    // and the allocation is tried again: a spare of its size, else spares given back again.
                    self.give_back(capacity,budget);
                    if counts.held.get()+capacity>budget && !counts.waiting.borrow().is_empty() {
                        self.drain();if self.take_spare(capacity) {return self.made(words,capacity);}self.give_back(capacity,budget);
                    }
                    if counts.held.get()+capacity>budget {counts.refused.set(true);}
                }
                counts.held.set(counts.held.get()+capacity);counts.most_held.set(counts.most_held.get().max(counts.held.get()));
            }
            counts.buffers.set(counts.buffers.get()+1);
        }
        self.made(words,capacity)
    }
    /// The value of a new block, and in device mode the counts of its buffer.
    fn made(&self,words:Vec<f32>,capacity:usize)->Held {
        let (counts,len)=(&self.counts,words.len());
        if counts.device.get() {
            counts.buffers.set(counts.buffers.get()+1);
            let (w,b)=counts.pages.get();let now=(w+capacity,b+1);counts.pages.set(now);if now.0>counts.most_pages.get().0 {counts.most_pages.set(now);}
        }
        Held {block:Rc::new(Block {words:RefCell::new(words),counts:self.counts.clone(),capacity}),start:0,len}
    }
    /// A spare of the pool of exactly `capacity` words (`Buffer::pooled`), if there is one.
    fn take_spare(&self,capacity:usize)->bool {
        let mut spares=self.counts.spares.borrow_mut();
        match spares.get_mut(&capacity) {Some(n) if *n>0=>{*n-=1;true},_=>false}
    }
    /// The pool gives spares back, the largest first, while a new buffer of `capacity` words passes `budget`.
    fn give_back(&self,capacity:usize,budget:usize) {
        let counts=&self.counts;let mut spares=counts.spares.borrow_mut();
        while counts.held.get()+capacity>budget {
            let Some((&size,n))=spares.iter_mut().rev().find(|(_,n)|**n>0) else {break};
            *n-=1;counts.held.set(counts.held.get()-size);
        }
    }
    /// From now, a new buffer that passes `budget` words held by the device makes the device free what it can, as
    /// the backend does; one that still passes it is counted (`Meter::refused`). None: no budget.
    pub(super) fn budget(&self,budget:Option<usize>) {self.counts.budget.set(budget);self.counts.refused.set(false);}
    /// Whether a new buffer passed the budget since `Meter::budget`.
    pub(super) fn refused(&self)->bool {self.counts.refused.get()}
    /// A meter that keeps the buffers of the device allocator as the gpu crate does inside a resident queue: a
    /// freed buffer waits for the next drain (a read or a fill: both wait for the stream), then it is a spare that a
    /// value of the same capacity takes again. No spare is ever given back here (no pool limit).
    pub(super) fn device()->Self {let meter=Self::default();meter.counts.device.set(true);meter}
    /// What a wait for the stream does: every freed buffer becomes a spare.
    fn drain(&self) {let mut spares=self.counts.spares.borrow_mut();for capacity in self.counts.waiting.borrow_mut().drain(..) {*spares.entry(capacity).or_default()+=1;}}
    /// Start a step as a device with no spare would: drain, give every spare back, and count the most held words and
    /// the buffers made from now. It gives the held words now (the buffers of what is alive).
    pub(super) fn start_device_step(&self)->usize {
        self.drain();let counts=&self.counts;
        let spare=std::mem::take(&mut *counts.spares.borrow_mut()).iter().map(|(capacity,n)|capacity*n).sum::<usize>();
        counts.held.set(counts.held.get()-spare);counts.most_held.set(counts.held.get());counts.buffers.set(0);
        counts.most_pages.set(counts.pages.get());counts.held.get()
    }
    /// The live device buffers now: their words in whole pages and their count.
    pub(super) fn pages(&self)->(usize,usize) {self.counts.pages.get()}
    /// The most words of live device buffers at one time since `start_device_step`, and their count then.
    pub(super) fn most_pages(&self)->(usize,usize) {self.counts.most_pages.get()}
    /// The most words that the device held at one time since `start_device_step`, and the buffers made since then.
    pub(super) fn device_most(&self)->(usize,usize) {(self.counts.most_held.get(),self.counts.buffers.get())}
    fn get(value:&Held)->Vec<f32> {value.block.words.borrow()[value.start..value.start+value.len].to_vec()}
    pub(super) fn live(&self)->usize {self.counts.live.get()}
    /// The most words are counted from now.
    pub(super) fn restart(&self) {self.counts.most.set(self.counts.live.get());}
    pub(super) fn most(&self)->usize {self.counts.most.get()}
    /// The unit launches (`rows` false) or the launches of the rows entry (`Compute::block_rows`) of `op` so far.
    pub(super) fn launched(&self,rows:bool,op:Op)->usize {self.launches.borrow().get(&(rows,op as u32)).copied().unwrap_or(0)}
}
impl Compute for Meter {
    type Value=Held;type Pass=();
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Held,_:usize)->Result<Option<Vec<u32>>> {Ok(None)}
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,_:Shape,_:[Option<&Held>;4],_:usize,_:f32)->Result<Held> {Err(NONE.into())}
    fn latent_absorption(&self,_:&str,_:Shape,_:&Held,_:&Held,_:&Held,_:usize)->Result<Held> {Err(NONE.into())}
    fn latent_absorption_rows(&self,_:&str,_:prompt_ops::Shape,_:&Held,_:&Held,_:&Held)->Result<Held> {Err(NONE.into())}
    fn latent_absorption_chats(&self,_:&str,_:prompt_ops::Shape,_:&[ChatGather<'_,Held>],_:usize,_:&Held,_:&Held)->Result<Option<Held>> {Ok(None)}
    fn upload(&self,v:&[f32])->Result<Held> {Host.upload(v).map(|v|self.make(v))}
    fn read(&self,v:&Held)->Result<Vec<f32>> {self.drain();Ok(Self::get(v))}
    fn select_known(&self,_:&Held,_:&Held,_:&[f32])->Result<()> {Ok(())}
    fn prompt_end(&self)->Result<()> {Ok(())}
    fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,_:&CheckpointAdmission,_:usize)->Result<()> {Ok(())}
    fn words(&self,v:&Held)->usize {v.len}
    fn range(&self,v:&Held,start:usize,len:usize)->Result<Held> {
        need(start.checked_add(len).is_some_and(|end|end<=v.len),"a range of the meter is outside its value")?;
        Ok(Held {block:v.block.clone(),start:v.start+start,len})
    }
    fn join(&self,parts:&[&Held])->Result<Held> {Ok(self.make(parts.iter().flat_map(|part|Self::get(part)).collect()))}
    fn zeros(&self,len:usize)->Result<Held> {Ok(self.make(vec![0.0;len]))}
    fn operation(&self,s:Shape,x:[Option<&Held>;4],w:&[f32])->Result<Held> {
        *self.launches.borrow_mut().entry((false,s.op as u32)).or_default()+=1;
        let x=x.map(|v|v.map(Self::get));Host.operation(s,[x[0].as_ref(),x[1].as_ref(),x[2].as_ref(),x[3].as_ref()],w).map(|v|self.make(v))
    }
    fn block_rows(&self,s:Shape,rows:usize,shared:u32,x:[Option<&Held>;4],w:&[f32])->Result<Option<Held>> {
        *self.launches.borrow_mut().entry((true,s.op as u32)).or_default()+=1;
        let x=x.map(|v|v.map(Self::get));Ok(Host.block_rows(s,rows,shared,[x[0].as_ref(),x[1].as_ref(),x[2].as_ref(),x[3].as_ref()],w)?.map(|v|self.make(v)))
    }
    fn prompt(&self,s:prompt_ops::Shape,x:[Option<&Held>;5],w:&[f32])->Result<Held> {
        need(!(s.op==prompt_ops::Op::Wrapped && spark_gpu_probe::block_ops::is_block(s.inner.op)),"a block arm runs in its own entry, not in the general prompt entry")?;
        let x=x.map(|v|v.map(Self::get));Host.prompt(s,[x[0].as_ref(),x[1].as_ref(),x[2].as_ref(),x[3].as_ref(),x[4].as_ref()],w).map(|v|self.make(v))
    }
    fn round(&self,x:&Held)->Result<Held> {Host.round(&Self::get(x)).map(|v|self.make(v))}
    /// In place, as on a device: no new value.
    fn rows_fill(&self,target:&mut Held,offset:usize,part:&Held,expected:usize)->Result<()> {
        self.drain();
        need(part.len==expected && offset.checked_add(expected).is_some_and(|end|end<=target.len),"a fill of the meter is outside its value")?;
        let (part,at)=(Self::get(part),target.start+offset);target.block.words.borrow_mut()[at..at+expected].copy_from_slice(&part);Ok(())
    }
    fn norm(&self,_:&str,_:NormShape,_:&Held,_:&[f32])->Result<Held> {Err(NONE.into())}
    fn begin(&self,_:(usize,usize),_:&crate::layer_operands::ResidualBlockOperands,_:&Held)->Result<((),Held)> {Err(NONE.into())}
    fn finish(&self,_:(),_:&Held)->Result<Held> {Err(NONE.into())}
    fn mx(&self,_:&Mxfp8Projection<'_>,_:&Held)->Result<Held> {Err(NONE.into())}
    fn mean(&self,_:&Held,_:u32,_:u32)->Result<Held> {Err(NONE.into())}
    fn rows_norm(&self,_:&str,_:NormShape,_:&Held,_:usize,_:&[f32])->Result<Held> {Err(NONE.into())}
    fn rows_begin(&self,_:(usize,usize),_:&crate::layer_operands::ResidualBlockOperands,_:&Held,_:usize)->Result<((),Held)> {Err(NONE.into())}
    fn rows_finish(&self,_:(),_:&Held,_:usize)->Result<Held> {Err(NONE.into())}
}
/// A chat through `driver` from a clean state, cut into steps of `cut` rows. It gives, for each token, the bits of
/// the stream after each visit and of the capped logits, then the top row; and the state after the chat.
fn chat<C:Compute<Value=Vec<f32>>,W:Weights<C>>(driver:&Driver<'_,C,W>,tokens:&[u32],cut:&[usize])->(Vec<Vec<u32>>,State<Vec<f32>>) {
    let d=driver.run.description;let (width,head)=(d.hidden as usize,d.head_rows as usize);
    let (mut state,mut at,mut out)=(driver.clean().unwrap(),0,Vec::new());
    for &count in cut {
        let mut streams=Vec::new();
        let step=driver.step_rows_seen(&mut [ChatStep {state:&mut state,tokens:&tokens[at..at+count],wanted:Wanted::All,logits:true}],|_,stream|{streams.push(stream.clone());Ok(())}).unwrap();
        let logits=step.logits.unwrap();
        for row in 0..count {
            let mut words=streams.iter().flat_map(|stream|bits(&stream[row*width..(row+1)*width])).collect::<Vec<_>>();
            words.extend(bits(&logits[row*head..(row+1)*head]));words.push(step.tops[row]);out.push(words);
        }
        at+=count;
    }
    assert_eq!((at,state.position),(tokens.len(),tokens.len() as u64));(out,state)
}
/// What a state holds, by position and not by offset: for each visit the key row and the value row of each position
/// that the last token saw, then the four rings. Two drivers with two step row limits hold a chat in planes of two
/// sizes, and they give the same list.
fn by_position<C:Compute<Value=Vec<f32>>,W:Weights<C>>(driver:&Driver<'_,C,W>,state:&State<Vec<f32>>)->Vec<u32> {
    let (mut out,last)=(Vec::new(),state.position-1);
    for (layout,visit) in driver.layouts().iter().zip(&state.visits) {
        for position in last+1-layout.live(last).unwrap() as u64..=last {
            let row=layout.slot(position).unwrap()*layout.row_words;
            out.extend(bits(&visit.keys[row..row+layout.row_words]));out.extend(bits(&visit.values[row..row+layout.row_words]));
        }
        for ring in &visit.rings {out.extend(bits(ring));}
    }
    out
}
/// Gate G4 of the rows work. The reference is the chat on the token path of a driver for one row a step: its
/// rings hold the window and no more, and they go round (windows of 4 and 3 tokens, 12 tokens). A driver for more
/// rows a step has larger rings, and a small byte limit cuts the part over keys into groups of rows.
#[test]
fn the_byte_limit_over_keys_and_the_extra_ring_rows_change_no_word() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(12);
        let one=Driver::new(&Host,&fixture,d,16).unwrap();
        let (mut state,mut words)=(one.clean().unwrap(),Vec::new());
        for &id in &tokens {
            let mut row=Vec::new();
            let step=one.step_seen(&mut state,id,|_,stream|{row.extend(bits(stream));Ok(())}).unwrap();
            row.extend(bits(&step.logits));row.push(step.top);words.push(row);
        }
        let kept=by_position(&one,&state);
        // The bytes of one row over keys at the largest plane of a driver for 12 rows a step. A limit of two such
        // rows gives groups of two rows there, and larger groups in a smaller plane.
        let wide=Driver::with_limits(&Host,&fixture,d,16,Limits {step_rows:12,key_bytes:1<<30}).unwrap();
        let row=d.visits.iter().map(|visit|{let a=d.layers[visit.layer as usize].attention;4*a.query_heads as usize*(2*wide.layouts()[visit.slot as usize].wide().rows+a.relative.extent as usize)}).max().unwrap();
        let (mut compared,mut pairs)=(0,0);
        for (step_rows,key_bytes,cut) in [(12,1<<30,vec![12]),(12,1,vec![12]),(12,2*row+1,vec![12]),(12,2*row+1,vec![2,3,4,3]),(5,1<<30,vec![5,5,2]),(5,1,vec![5,5,2]),(5,3*row,vec![4,5,3]),(1,1,vec![1;12])] {
            let driver=Driver::with_limits(&Host,&fixture,d,16,Limits {step_rows,key_bytes}).unwrap();
            assert!(driver.layouts().iter().all(|layout|layout.key_bytes==key_bytes && layout.rows==if layout.ring {layout.window} else {16} && layout.wide().rows==if layout.ring {layout.window+step_rows-1} else {16}));
            let (rows,state)=chat(&driver,&tokens,&cut);
            assert_eq!(rows,words,"{name}: step row limit {step_rows}, {key_bytes} bytes over keys, cut {cut:?}");
            assert_eq!(by_position(&driver,&state),kept,"{name}: the state by position, step row limit {step_rows}, {key_bytes} bytes over keys, cut {cut:?}");
            compared+=rows.iter().map(Vec::len).sum::<usize>()+kept.len();pairs+=1;
        }
        println!("G4 {name}: {compared} words equal the token path, for {pairs} pairs of a step row limit and a byte limit over keys");
    }
}
/// The rule of a step that fails (`Driver::step_rows_seen`): no position moves, nothing waits in a state, and the
/// same step, or its rows without one chat, runs again on the same states with the same words.
/// Three cases. Before a ring goes round: steps of 3 and 2 rows, then the step that fails, 4 and 3 rows. Where a
/// ring goes round (rings of window + 4 rows, 8 and 7 here): two steps of 3 and 2 rows, then the step that fails
/// writes positions 6 to 8 of the first chat. And the largest step there, 5 rows of one chat from position 6: its
/// last row writes the plane row of the oldest key that the token before the step saw.
#[test]
fn a_step_of_rows_that_fails_runs_again_with_the_same_words() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
      for (step_rows,before_it,failing) in [(8usize,vec![[3usize,2usize]],[4usize,3usize]),(5,vec![[3,2],[3,2]],[3,2]),(5,vec![[3,2],[3,2]],[5,0])] {
        let d=&fixture.description;let tokens=fixture.chat(16);let other=tokens.iter().rev().copied().collect::<Vec<_>>();
        let driver=Driver::with_limits(&Host,&fixture,d,16,Limits {step_rows,key_bytes:1<<30}).unwrap();
        // One step of two chats: `counts` rows of each (0: the chat is not in the step). The caller refuses the
        // stream after visit `fail`. It gives the bits of every stream, of the logits, and the top rows.
        let step=|states:&mut [State<Vec<f32>>;2],counts:[usize;2],fail:Option<usize>|->Result<Vec<u32>> {
            let [first,second]=states;let (a,b,mut words,mut chats)=(first.position as usize,second.position as usize,Vec::new(),Vec::new());
            if counts[0]>0 {chats.push(ChatStep {state:first,tokens:&tokens[a..a+counts[0]],wanted:Wanted::All,logits:true});}
            if counts[1]>0 {chats.push(ChatStep {state:second,tokens:&other[b..b+counts[1]],wanted:Wanted::Last,logits:true});}
            let out=driver.step_rows_seen(&mut chats,|place,stream|{if fail==Some(place) {return Err("the caller refuses this visit".into());}words.extend(bits(stream));Ok(())})?;
            words.extend(bits(&out.logits.unwrap()));words.extend(out.tops);Ok(words)
        };
        let fresh=||[driver.clean().unwrap(),driver.clean().unwrap()];
        let places=|states:&[State<Vec<f32>>;2]|[by_position(&driver,&states[0]),by_position(&driver,&states[1])];
        let start=|states:&mut [State<Vec<f32>>;2]|before_it.iter().map(|&counts|step(states,counts,None).unwrap()).collect::<Vec<_>>();
        let [first,second]=failing;let at=[before_it.iter().map(|counts|counts[0]).sum::<usize>() as u64,before_it.iter().map(|counts|counts[1]).sum::<usize>() as u64];
        // The step that fails goes over the end of a ring of the first chat, or not, as the case says.
        let ring=driver.layouts().iter().filter(|layout|layout.ring).map(|layout|layout.wide().rows as u64).min().unwrap();
        assert_eq!(at[0]+first as u64>ring,step_rows==5,"{name}: the case and the ring");
        // Two twins never fail: one takes the last step whole, one takes it a chat at a time.
        let (mut whole,mut apart)=(fresh(),fresh());
        let before=start(&mut whole);assert_eq!(start(&mut apart),before);
        let after=step(&mut whole,failing,None).unwrap();
        let alone=if second>0 {vec![step(&mut apart,[first,0],None).unwrap(),step(&mut apart,[0,second],None).unwrap()]} else {vec![step(&mut apart,failing,None).unwrap()]};
        assert_eq!(places(&whole),places(&apart),"{name}: a chat alone and beside another chat");
        let mut failed=0;
        for fail in [0,d.visits.len()/2,d.visits.len()-1] {
            for again in [true,false] {
                let mut held=fresh();assert_eq!(start(&mut held),before);let kept=places(&held);
                assert!(step(&mut held,failing,Some(fail)).unwrap_err().contains("refuses this visit"));failed+=1;
                assert_eq!([held[0].position,held[1].position],at,"{name}: a step that failed at visit {fail} moved a chat");
                // A step of fewer rows than the limit writes no plane row that the token before it saw.
                if first<step_rows {assert_eq!(places(&held),kept,"{name}: a step that failed at visit {fail} changed a row that a chat had");}
                assert!(held.iter().all(|state|state.visits.iter().all(|visit|visit.next.iter().all(Option::is_none))),"{name}: a ring still waits");
                if again || second==0 {assert_eq!(step(&mut held,failing,None).unwrap(),after,"{name}: the same step after a failure at visit {fail}");}
                else {assert_eq!(vec![step(&mut held,[first,0],None).unwrap(),step(&mut held,[0,second],None).unwrap()],alone,"{name}: the rows of each chat alone after a failure at visit {fail}");}
                assert_eq!(([held[0].position,held[1].position],places(&held)),([at[0]+first as u64,at[1]+second as u64],places(&whole)),"{name}: the states after a failure at visit {fail}");
            }
        }
        println!("RERUN {name}, step row limit {step_rows}: {failed} failed steps of {first} and {second} rows of two chats from positions {at:?} (ring of {ring} rows) ran again with the same words");
      }
    }
}
/// A step that the driver cannot run is refused before any launch and before any value is made, and the text names
/// the chat: a state of a driver with another capacity, and a state with a ring that
/// still waits. A step row limit that one launch cannot take is refused when the driver is made.
#[test]
fn a_step_with_a_state_of_another_driver_is_refused_before_any_launch() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(12);let meter=Meter::default();
        let driver=Driver::with_limits(&meter,&fixture,d,16,Limits {step_rows:8,key_bytes:1<<30}).unwrap();
        // The driver of one row a step keeps states of the same sizes (window rings of the window): its state is taken.
        assert_eq!(Driver::new(&meter,&fixture,d,16).unwrap().state_words(),driver.state_words(),"{name}: one row a step");
        let others=[Driver::with_limits(&meter,&fixture,d,12,Limits {step_rows:8,key_bytes:1<<30}).unwrap()];
        for (which,other) in others.iter().enumerate() {
            let (mut own,mut wrong)=(driver.clean().unwrap(),other.clean().unwrap());
            let base=meter.live();meter.restart();meter.launches.borrow_mut().clear();
            let error=driver.step_rows(&mut [ChatStep {state:&mut own,tokens:&tokens[..3],wanted:Wanted::Last,logits:true},ChatStep {state:&mut wrong,tokens:&tokens[..2],wanted:Wanted::None,logits:false}]).unwrap_err();
            assert!(error.contains("chat 1 of the step") && error.contains("sizes of another driver"),"{name} driver {which}: {error}");
            assert!(meter.launches.borrow().is_empty() && meter.most()==base && meter.live()==base,"{name} driver {which}: a refused step ran a launch or made a value");
            assert_eq!((own.position,wrong.position),(0,0));
            // The state of this driver was not written: it goes on as a clean state does.
            let mut twin=driver.clean().unwrap();
            let tops=|state:&mut State<Held>|driver.step_rows(&mut [ChatStep {state,tokens:&tokens[..3],wanted:Wanted::All,logits:false}]).unwrap().tops;
            assert_eq!(tops(&mut own),tops(&mut twin));
        }
        // A ring that still waits: a state that a step left without `commit` or `discard`.
        let mut state=driver.clean().unwrap();state.visits[0].next[0]=Some(meter.make(vec![0.0;4]));
        let error=driver.step_rows(&mut [ChatStep {state:&mut state,tokens:&tokens[..1],wanted:Wanted::None,logits:false}]).unwrap_err();
        assert!(error.contains("chat 0 of the step") && error.contains("still waits"),"{name}: {error}");
        // One launch of rows takes `prompt_ops::MAX_ROWS` rows (and `block_rows::ROWS_MAX`, the same count).
        let limit=|step_rows:usize|Driver::with_limits(&meter,&fixture,d,16,Limits {step_rows,key_bytes:1<<30}).map(|_|());
        assert!(limit(prompt_ops::MAX_ROWS).is_ok());assert!(limit(prompt_ops::MAX_ROWS+1).unwrap_err().contains("above the rows of one launch"));
    }
}
/// The rings of the chats of one step are ranges of one value (`state::row_rings`). On a backend whose range is a
/// view, a chat that takes no later step keeps the rings of the chats that ended. `Driver::detach` gives the chat
/// rings of its own, and so does its next step. The words of the chat are the same in each case.
#[test]
fn a_chat_that_waits_keeps_the_rings_of_its_step_until_it_is_detached_or_steps() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(12);let meter=Meter::default();
        let driver=Driver::with_limits(&meter,&fixture,d,16,Limits {step_rows:8,key_bytes:1<<30}).unwrap();
        let rings=driver.layouts().iter().map(|l|2*(l.kv_taps*l.kv_channels+l.taps*l.channels)).sum::<usize>();
        // Four chats take two steps together. Then three of them end.
        let last=|| {
            let mut states=(0..4).map(|_|driver.clean().unwrap()).collect::<Vec<_>>();
            for at in 0..2 {
                let mut chats=states.iter_mut().enumerate().map(|(chat,state)|ChatStep {state,tokens:&tokens[chat+at..chat+at+1],wanted:Wanted::None,logits:false}).collect::<Vec<_>>();
                driver.step_rows(&mut chats).unwrap();
            }
            assert_eq!(meter.live(),4*driver.state_words(),"{name}: four chats hold four states");
            states.pop().unwrap()
        };
        let next=|state:&mut State<Held>|{let out=driver.step_rows(&mut [ChatStep {state,tokens:&tokens[5..6],wanted:Wanted::Last,logits:true}]).unwrap();(out.tops,Meter::get(&out.logits.unwrap()).iter().map(|v|v.to_bits()).collect::<Vec<_>>())};
        // The chat that waits holds its state and the rings of the three chats that ended.
        let mut waits=last();assert_eq!(meter.live(),driver.state_words()+3*rings,"{name}: the chat that waits keeps the rings of its step");
        driver.detach(&mut waits).unwrap();assert_eq!(meter.live(),driver.state_words(),"{name}: after `detach` the chat holds its state and no more");
        let detached=next(&mut waits);assert_eq!(meter.live(),driver.state_words());drop(waits);
        // Without `detach` the next step of the chat gives it new rings, and the words are the same.
        let mut steps=last();assert_eq!(meter.live(),driver.state_words()+3*rings);
        assert_eq!(next(&mut steps),detached,"{name}: a chat goes on with the same words with and without `detach`");
        assert_eq!(meter.live(),driver.state_words(),"{name}: the next step of a chat lets the rings of its last step go");
        println!("DETACH {name}: a chat of a step of 4 chats that waits keeps {} words of rings of the 3 chats that ended, beside its state of {} words; 0 after detach, 0 after its next step",3*rings,driver.state_words());
    }
}
/// The words that a step holds at one time, MEASURED with `Meter`, stay under `Driver::transient_words`, and a step
/// leaves no word alive but the states and what it gives back. Each step runs after one step of one row a chat,
/// so its rings are those that a step of rows leaves.
#[test]
fn the_values_of_a_step_stay_under_the_bound_of_the_driver() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(16);let (mut steps,mut least,mut largest)=(0,f64::MAX,0f64);
        for key_bytes in [1<<30,1] {
            let meter=Meter::default();
            let driver=Driver::with_limits(&meter,&fixture,d,16,Limits {step_rows:12,key_bytes}).unwrap();
            let run=|states:&mut [State<Held>],counts:&[usize],wanted:Wanted,logits:bool|{
                let mut chats=states.iter_mut().zip(counts).map(|(state,&count)|{let at=state.position as usize;ChatStep {state,tokens:&tokens[at..at+count],wanted,logits}}).collect::<Vec<_>>();
                driver.step_rows(&mut chats).unwrap()
            };
            // The rows of each chat of the step, the rows that run the end, and if their logits go back.
            let plans:[(&[usize],Wanted,bool);10]=[(&[1],Wanted::Last,true),(&[1],Wanted::None,false),(&[2],Wanted::All,true),(&[5],Wanted::None,false),(&[12],Wanted::Last,false),(&[12],Wanted::All,true),
                (&[3,1,2],Wanted::All,true),(&[1;8],Wanted::All,true),(&[1;8],Wanted::Last,false),(&[2;6],Wanted::None,false)];
            for (counts,wanted,logits) in plans {
                let mut states=counts.iter().map(|_|driver.clean().unwrap()).collect::<Vec<_>>();
                drop(run(&mut states,&vec![1;counts.len()],Wanted::None,false));
                let base=meter.live();assert_eq!(base,counts.len()*driver.state_words(),"{name}: the states after a step are not the size of clean states");
                meter.restart();
                let out=run(&mut states,counts,wanted,logits);
                let (rows,chats)=(counts.iter().sum::<usize>(),counts.len());
                let ends=match wanted {Wanted::None=>0,Wanted::Last=>chats,Wanted::All=>rows};let back=if logits {ends} else {0};
                let (most,bound)=(meter.most()-base,driver.transient_words(rows,chats,ends,back));
                assert!(most<=bound,"{name}: a step of {rows} rows of {chats} chats, {ends} ends, held {most} words at one time, above the bound {bound}");
                assert_eq!((out.tops.len(),out.logits.as_ref().map_or(0,|logits|logits.len)),(ends,back*d.head_rows as usize));
                assert_eq!(meter.live(),base+back*d.head_rows as usize,"{name}: a step left words alive");drop(out);
                let times=bound as f64/most as f64;steps+=1;least=least.min(times);largest=largest.max(times);
                println!("BOUND {name}, {key_bytes} bytes over keys: {rows} rows of {chats} chats, {ends} ends, {back} with logits: most {most} words at one time, bound {bound}, {times:.2} times");
            }
        }
        println!("BOUND {name}: {steps} steps under the bound; the bound is {least:.2} to {largest:.2} times the most words that a step held");
        // The end alone, for 5 rows: the last norm, the head product and the capped logits. This is the part of the
        // bound that grows with the head rows; on these checkpoints the head is small and a visit holds more.
        let meter=Meter::default();let driver=Driver::new(&meter,&fixture,d,16).unwrap();
        let (width,head)=(d.hidden as usize,d.head_rows as usize);
        let mixed=meter.make((0..5*width).map(|i|0.25+(i%7) as f32*0.125).collect());let base=meter.live();meter.restart();
        let (logits,tops)=driver.capped(&mixed,5).unwrap();
        let (most,part)=(meter.most()-base,5*(2*width+2*head+1));
        assert!(most<=part && logits.len==5*head && tops.len()==5,"{name}: the end of 5 rows held {most} words, above its part of the bound {part}");
        println!("BOUND {name}: the end of 5 rows held {most} words at one time, its part of the bound is {part}");
    }
}
/// The block arms that the driver sends through `rows::rows_operation`.
pub(super) const ROW_ARMS:[Op;9]=[Op::HeadClip,Op::RelBias,Op::ShortConv,Op::WindowScores,Op::SinkSoftmax,Op::WindowValues,Op::MixBlend,Op::GateBlend,Op::RouteFold];
/// With the rows entry of the arms (`Compute::block_rows`) a step of many rows runs an arm for all rows in one
/// launch. For each arm that the driver sends through `rows::rows_operation`: a step of 12 rows of one chat runs as
/// many rows launches as a step of one row runs unit launches, and no unit launch. In a step of two chats the arms
/// over keys run once for each chat, each with the plane of its chat. The meter refuses a block arm in the general
/// prompt entry, so no step here puts one there.
#[test]
fn a_step_of_rows_runs_each_block_arm_in_one_launch_of_the_rows_entry() {
    let arms=ROW_ARMS;
    let over_keys=[Op::RelBias,Op::WindowScores,Op::SinkSoftmax,Op::WindowValues];
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(16);let meter=Meter::default();
        let driver=Driver::with_limits(&meter,&fixture,d,16,Limits {step_rows:12,key_bytes:1<<30}).unwrap();
        // One step from clean states, `counts` rows of each chat: the unit launches and the rows launches of each arm.
        let step=|counts:&[usize]|{
            let mut states=counts.iter().map(|_|driver.clean().unwrap()).collect::<Vec<_>>();
            meter.launches.borrow_mut().clear();
            let mut chats=states.iter_mut().zip(counts).map(|(state,&count)|ChatStep {state,tokens:&tokens[..count],wanted:Wanted::None,logits:false}).collect::<Vec<_>>();
            driver.step_rows(&mut chats).unwrap();
            arms.map(|op|[false,true].map(|rows|meter.launched(rows,op)))
        };
        let (one,many,two)=(step(&[1]),step(&[12]),step(&[7,5]));
        for (arm,op) in arms.into_iter().enumerate() {
            assert!(one[arm][0]>0 && one[arm][1]==0,"{name}: a step of one row has no unit launch of {op:?}, or a rows launch");
            assert_eq!(many[arm],[0,one[arm][0]],"{name}: {op:?} in a step of 12 rows of one chat");
            assert_eq!(two[arm],[0,if over_keys.contains(&op) {2} else {1}*one[arm][0]],"{name}: {op:?} in a step of 7 and 5 rows of two chats");
        }
        println!("ROWS ENTRY {name}: a step of one row runs {} unit launches of the 9 arms; a step of 12 rows of one chat runs {} rows launches of them and {} unit launches",
            one.iter().map(|arm|arm[0]).sum::<usize>(),many.iter().map(|arm|arm[1]).sum::<usize>(),many.iter().map(|arm|arm[0]).sum::<usize>());
    }
}
/// The device form of the bound test. The meter keeps buffers as the device allocator does inside a resident queue
/// (`Meter::device`): a value is whole pages, a freed buffer waits for a stream wait, a spare waits for a value of its
/// size. A step starts with no spare and no waiting buffer.
/// 1. The live device buffers of the step stay under `Driver::device_words` (words) and `transient_buffers` (count).
/// 2. With the budget of the device set to what was held before the step plus `device_words`, no allocation of the
///    step passes it: the waiting buffers and the spares that the device holds above it are freed as the backend
///    frees them. With the budget one page under what the live buffers needed, an allocation passes it: the live
///    buffers are what a step needs, and the model of the allocator is not looser than the device.
#[test]
fn the_device_buffers_of_a_step_stay_under_the_bound_of_the_driver() {
    for (name,fixture) in [("small",Fixture::small()),("wide",Fixture::wide())] {
        let d=&fixture.description;let tokens=fixture.chat(64);let (mut steps,mut least,mut largest,mut held_above)=(0,f64::MAX,0f64,0usize);
        for (step_rows,key_bytes) in [(12,1<<30),(12,1),(48,1<<30),(48,1)] {
            let meter=Meter::device();
            let driver=Driver::with_limits(&meter,&fixture,d,d.max_positions.min(64),Limits {step_rows,key_bytes}).unwrap();
            let plans:Vec<(Vec<usize>,Wanted,bool)>=if step_rows==12 {vec![(vec![1],Wanted::Last,true),(vec![1],Wanted::None,false),(vec![2],Wanted::All,true),(vec![5],Wanted::None,false),(vec![12],Wanted::Last,false),(vec![12],Wanted::All,true),
                (vec![3,1,2],Wanted::All,true),(vec![1;8],Wanted::All,true),(vec![1;8],Wanted::Last,false),(vec![2;6],Wanted::None,false)]}
                else {vec![(vec![40],Wanted::All,true),(vec![46],Wanted::Last,true),(vec![6;8],Wanted::All,true),(vec![1;48],Wanted::Last,true),(vec![20,2,1,3],Wanted::None,false)]};
            for (counts,wanted,logits) in plans {
                let (rows,chats)=(counts.iter().sum::<usize>(),counts.len());
                let ends=match wanted {Wanted::None=>0,Wanted::Last=>chats,Wanted::All=>rows};let back=if logits {ends} else {0};
                let (bound,buffers)=(driver.device_words(rows,chats,ends,back),driver.transient_buffers(rows,chats));
                // One run of the step: the states take a first token, then the device starts the step clean.
                let run=|budget:Option<usize>|{
                    let mut states=counts.iter().map(|_|driver.clean().unwrap()).collect::<Vec<_>>();
                    let mut first=states.iter_mut().map(|state|ChatStep {state,tokens:&tokens[0..1],wanted:Wanted::None,logits:false}).collect::<Vec<_>>();
                    drop(driver.step_rows(&mut first).unwrap());
                    let base=meter.start_device_step();let before=meter.pages();meter.budget(budget.map(|above|base+above));
                    let mut chats=states.iter_mut().zip(&counts).map(|(state,&count)|{let at=state.position as usize;ChatStep {state,tokens:&tokens[at..at+count],wanted,logits}}).collect::<Vec<_>>();
                    drop(driver.step_rows(&mut chats).unwrap());
                    let ((pages,live),(held,_))=(meter.most_pages(),meter.device_most());let refused=meter.refused();meter.budget(None);
                    (pages-before.0,live-before.1,held-base,refused)
                };
                let (pages,live,held,_)=run(None);
                assert!(pages<=bound && live<=buffers,"{name}, {step_rows} rows a step, {key_bytes} bytes over keys, chats {counts:?}: live device buffers {live} of {pages} words, bound {buffers} buffers of {bound} words");
                assert!(!run(Some(bound)).3,"{name}, chats {counts:?}: a buffer passed the budget of the bound {bound}");
                assert!(run(Some(pages-1024)).3,"{name}, chats {counts:?}: no buffer passed a budget one page under the live buffers {pages}");
                let times=bound as f64/pages as f64;steps+=1;least=least.min(times);largest=largest.max(times);held_above=held_above.max(held.saturating_sub(pages));
                println!("DEVICE {name}, {step_rows} rows a step, {key_bytes} bytes over keys: {rows} rows of {chats} chats, {ends} ends, {back} with logits: live buffers at most {pages} words in {live} buffers; with no budget the device held {held} words; bound {bound} words, {buffers} buffers; {times:.2} times");
            }
        }
        println!("DEVICE {name}: {steps} steps under the bound; the bound is {least:.2} to {largest:.2} times the live device words of a step; with no budget the device held up to {held_above} words more than the live buffers");
    }
}
