//! The text service on the two Sparks of a cabled pair: the scheduler runs on the Spark of half 0, and the Spark of
//! half 1 repeats each target call that changes a chat, in the same order, on its own service.
//!
//! Both Sparks hold one service over the split weights (`super::Split`). The cut products of a step trade words
//! between the two Sparks, so both must run the same target steps on the same chats. Half 0 wraps its service in a
//! `Mirror`: each target call of the scheduler that opens, steps, verifies, accepts, forks, cuts or ends a chat goes
//! first to half 1 on the control wire, then runs on half 0.
//!
//! Speculative drafting (`--mtp N`) runs ONLY on half 0 with `Split::bypass` set (zero RDMA trades during propose).
//! Half 1 never loads or executes `Drafter`: it mirrors only the resulting target verify, accept, and step calls so
//! timing-based draft depth choices on half 0 can never diverge from half 1, guaranteeing bit-exact `0` diffs.
//!
//! No line here names a model, a size or a limit: a message holds what the call holds.
use crate::{Result,bounded_decoder::Compute,loop_decoder::{experts::Phase,mix::Weights,service::{Chat,FollowBatchRow,Service}}};
use spark_engine::draft::{ModelProposal,Target};
use spark_serve::{BatchError,BatchOutput,BatchRows,DraftRows,Engine,Predictions,Session,protocol::Request};
use std::{cell::{Cell,RefCell},collections::HashMap,io::{Read,Write},net::TcpStream,time::Duration};

/// The calls on the wire. A message is `[call, words of the call..., tokens...]`, each word 8 bytes.
mod call {
    pub const OPEN:u64=1;pub const OPEN_FOR:u64=2;pub const CLOSE:u64=3;pub const FORK:u64=4;pub const FORK_PREFIX:u64=5;
    pub const STEP:u64=6;pub const ADVANCE:u64=7;pub const PREFILL:u64=8;pub const PREFILL_ADVANCE:u64=9;pub const SLICE:u64=10;
    pub const SETTLE:u64=11;pub const ADMISSION:u64=12;pub const RESERVE_DRAFT:u64=13;pub const NO_DRAFT:u64=14;pub const DRAFT_OVER:u64=15;
    pub const BATCH:u64=16;pub const COHORT:u64=17;pub const STOP:u64=18;pub const CONFIGURE:u64=19;pub const RETENTION:u64=20;
    pub const RELEASE:u64=21;pub const AUTOMATIC:u64=22;pub const SPARE_SCOPE:u64=23;pub const RETAIN_DRAFT:u64=24;pub const RETIRE:u64=25;
    pub const TRUNCATE:u64=26;pub const LOGITS:u64=27;pub const FINISH:u64=28;
    pub const ACCEPT_DRAFT:u64=29;pub const FLUSH_ACCEPTS:u64=30;
}
/// The kinds of a batch call.
const PREFILL_BATCH:u64=0;const STEP_BATCH:u64=1;

/// RAII guard setting `Split::bypass` during rank-0-only drafter calls so no drafter weight read ever triggers RDMA.
struct BypassGuard<'a> {cell:Option<&'a Cell<bool>>,was:bool}
impl<'a> BypassGuard<'a> {
    fn enter(cell:Option<&'a Cell<bool>>)->Self {let was=cell.is_some_and(|c|c.replace(true));Self {cell,was}}
}
impl Drop for BypassGuard<'_> {fn drop(&mut self) {if let Some(c)=self.cell {c.set(self.was);}}}

/// One end of the control wire: messages of 8-byte words, each after its word count.
pub struct Wire {stream:TcpStream}
impl Wire {
    pub fn new(stream:TcpStream)->Result<Self> {
        stream.set_nodelay(true).map_err(|e|format!("tp2 mirror wire: {e}"))?;
        stream.set_read_timeout(None).map_err(|e|format!("tp2 mirror wire: {e}"))?;
        Ok(Self {stream})
    }
    fn send(&mut self,words:&[u64])->Result<()> {
        let mut bytes=Vec::with_capacity(8*(words.len()+1));
        bytes.extend_from_slice(&(words.len() as u64).to_le_bytes());
        for word in words {bytes.extend_from_slice(&word.to_le_bytes());}
        self.stream.write_all(&bytes).map_err(|e|format!("tp2 mirror wire: {e}"))
    }
    /// The next message. None: the other Spark closed the wire.
    fn receive(&mut self)->Result<Option<Vec<u64>>> {
        let mut head=[0u8;8];
        match self.stream.read_exact(&mut head) {
            Ok(())=>{},
            Err(e) if e.kind()==std::io::ErrorKind::UnexpectedEof=>return Ok(None),
            Err(e)=>return Err(format!("tp2 mirror wire: {e}")),
        }
        let words=usize::try_from(u64::from_le_bytes(head)).map_err(|_|"tp2 mirror wire: a message exceeds this host".to_owned())?;
        let mut bytes=vec![0u8;words.checked_mul(8).ok_or("tp2 mirror wire: a message exceeds this host")?];
        self.stream.read_exact(&mut bytes).map_err(|e|format!("tp2 mirror wire: {e}"))?;
        Ok(Some(bytes.as_chunks::<8>().0.iter().map(|b|u64::from_le_bytes(*b)).collect()))
    }
}
#[cfg(test)]
impl Wire {
    pub(super) fn send_for_test(&mut self,words:&[u64])->Result<()> {self.send(words)}
    pub(super) fn receive_for_test(&mut self)->Result<Option<Vec<u64>>> {self.receive()}
}
fn message(call:u64,words:&[u64],tokens:&[u32])->Vec<u64> {
    let mut m=Vec::with_capacity(1+words.len()+tokens.len());
    m.push(call);m.extend_from_slice(words);m.extend(tokens.iter().map(|&t|u64::from(t)));m
}
fn output_word(output:BatchOutput)->u64 {match output {BatchOutput::All=>0,BatchOutput::Last=>1,BatchOutput::None=>2}}
fn output_of(word:u64)->Result<BatchOutput> {match word {0=>Ok(BatchOutput::All),1=>Ok(BatchOutput::Last),2=>Ok(BatchOutput::None),_=>Err(format!("tp2 mirror: no batch output {word}"))}}

/// The sending end of half 0: the wire, the chat numbers, and the first failure of the wire.
pub struct Link {wire:RefCell<Wire>,next:Cell<u64>,failed:RefCell<Option<String>>}
impl Link {
    pub fn new(wire:Wire)->Self {Self {wire:RefCell::new(wire),next:Cell::new(0),failed:RefCell::new(None)}}
    /// Send one message. A failed wire is kept as the fatal error of the service (`Engine::fatal_error`): half 1
    /// can no longer follow, so no call after it is sent.
    fn send(&self,words:Vec<u64>) {
        if self.failed.borrow().is_some() {return;}
        if let Err(error)=self.wire.borrow_mut().send(&words) {*self.failed.borrow_mut()=Some(error);}
    }
    fn id(&self)->u64 {let id=self.next.get()+1;self.next.set(id);id}
    /// Tell half 1 that the service ends.
    pub fn stop(&self) {self.send(message(call::STOP,&[],&[]));}
}
/// The service of half 0 behind the scheduler.
pub struct Mirror<'e,'a,C:Compute,W:Weights<C>> {inner:&'e Service<'a,C,W>,link:&'e Link,bypass:Option<&'e Cell<bool>>,verify_row_cap:usize}
impl<'e,'a,C:Compute,W:Weights<C>> Mirror<'e,'a,C,W> {
    pub fn with_draft(inner:&'e Service<'a,C,W>,link:&'e Link,bypass:&'e Cell<bool>,verify_row_cap:usize)->Self {Self {inner,link,bypass:Some(bypass),verify_row_cap}}
}
/// One chat of half 0: the chat of its service and its number on the wire.
pub struct MirrorSession<'s,'a,C:Compute,W:Weights<C>> {inner:Chat<'s,'a,C,W>,id:u64,link:&'s Link,bypass:Option<&'s Cell<bool>>}
impl<C:Compute,W:Weights<C>> Drop for MirrorSession<'_,'_,C,W> {fn drop(&mut self) {self.link.send(message(call::CLOSE,&[self.id],&[]));}}
impl<C:Compute,W:Weights<C>> MirrorSession<'_,'_,C,W> {
    fn send(&self,call:u64,words:&[u64],tokens:&[u32]) {
        let mut head=Vec::with_capacity(1+words.len());head.push(self.id);head.extend_from_slice(words);
        self.link.send(message(call,&head,tokens));
    }
}
/// Open a chat on both Sparks: the message first, then the call of half 0. A call that fails on half 0 closes the
/// chat on half 1 too.
fn opened<'s,'a,C:Compute,W:Weights<C>>(link:&'s Link,bypass:Option<&'s Cell<bool>>,call:u64,words:&[u64],open:impl FnOnce()->Result<Chat<'s,'a,C,W>>)->Result<MirrorSession<'s,'a,C,W>> {
    let id=link.id();
    let mut head=vec![id];head.extend_from_slice(words);
    link.send(message(call,&head,&[]));
    match open() {
        Ok(inner)=>Ok(MirrorSession {inner,id,link,bypass}),
        Err(error)=>{link.send(message(call::CLOSE,&[id],&[]));Err(error)},
    }
}
impl<C:Compute,W:Weights<C>> Target for MirrorSession<'_,'_,C,W> {
    fn reject_unmapped_greedy(&self)->bool {Session::reject_unmapped_greedy(&self.inner)}
    fn checkpoint(&self)->usize {Target::checkpoint(&self.inner)}
    fn truncate(&mut self,position:usize)->Result<()> {
        if let Some((rows,keep,defer))=self.inner.pending_verify_action(position) {
            self.send(call::ACCEPT_DRAFT,&[rows as u64,keep as u64,u64::from(defer)],&[]);
        }
        Target::truncate(&mut self.inner,position)
    }
    fn step(&mut self,token:u32)->Result<Vec<f32>> {<Self as Session>::step(self,token)}
    fn native_draft_policy(&self)->bool {Target::native_draft_policy(&self.inner)}
    fn max_verify_rows(&self)->usize {Target::max_verify_rows(&self.inner)}
    fn model_draft_width(&self)->usize {Target::model_draft_width(&self.inner)}
    fn defer_model_draft(&self,limit:usize)->Option<usize> {Target::defer_model_draft(&self.inner,limit)}
    fn propose_model(&mut self,history:&[u32],anchor:u32,limit:usize)->Result<Option<ModelProposal>> {
        let _guard=BypassGuard::enter(self.bypass);
        Target::propose_model(&mut self.inner,history,anchor,limit)
    }
}
impl<C:Compute,W:Weights<C>> Session for MirrorSession<'_,'_,C,W> {
    fn configure_request(&mut self,request:&Request)->Result<()> {self.send(call::CONFIGURE,&[],&[]);self.inner.configure_request(request)}
    fn cached_prompt_tokens(&self,prompt_tokens:usize)->Option<usize> {self.inner.cached_prompt_tokens(prompt_tokens)}
    fn set_cache_retention(&mut self,retain:bool) {self.send(call::RETENTION,&[u64::from(retain)],&[]);self.inner.set_cache_retention(retain)}
    fn disable_model_draft(&mut self) {self.send(call::NO_DRAFT,&[],&[]);self.inner.disable_model_draft()}
    fn reject_unmapped_greedy(&self)->bool {Session::reject_unmapped_greedy(&self.inner)}
    fn draft_over(&mut self)->bool {self.send(call::DRAFT_OVER,&[],&[]);self.inner.draft_over()}
    fn release_replay_snapshot(&mut self) {self.send(call::RELEASE,&[],&[]);self.inner.release_replay_snapshot()}
    fn reserve_model_draft(&mut self,tokens:usize) {self.send(call::RESERVE_DRAFT,&[tokens as u64],&[]);self.inner.reserve_model_draft(tokens)}
    fn admission_tokens(&mut self,tokens:usize) {self.send(call::ADMISSION,&[tokens as u64],&[]);self.inner.admission_tokens(tokens)}
    fn automatic_draft(&mut self,prompt:usize,remaining:usize) {self.send(call::AUTOMATIC,&[prompt as u64,remaining as u64],&[]);self.inner.automatic_draft(prompt,remaining)}
    fn draft_target(&mut self)->Option<&mut dyn spark_engine::draft::Target> {if self.inner.has_draft() {Some(self)} else {None}}
    fn memory_bytes(&self)->Option<(usize,usize)> {self.inner.memory_bytes()}
    fn draft_model_memory_bytes(&self)->Option<(usize,usize)> {self.inner.draft_model_memory_bytes()}
    fn step(&mut self,token:u32)->Result<Vec<f32>> {self.send(call::STEP,&[u64::from(self.inner.has_draft())],&[token]);Session::step(&mut self.inner,token)}
    fn advance(&mut self,token:u32)->Result<()> {self.send(call::ADVANCE,&[u64::from(self.inner.has_draft())],&[token]);self.inner.advance(token)}
    fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>> {self.send(call::PREFILL,&[u64::from(self.inner.has_draft())],tokens);self.inner.prefill(tokens)}
    fn prefill_advance(&mut self,tokens:&[u32])->Result<()> {self.send(call::PREFILL_ADVANCE,&[u64::from(self.inner.has_draft())],tokens);self.inner.prefill_advance(tokens)}
    fn read_ahead(&self,tokens:&[u32]) {self.inner.read_ahead(tokens)}
    fn prefill_chunk_size(&self,prompt_end:usize)->usize {self.inner.prefill_chunk_size(prompt_end)}
    fn prefill_chunk_min(&self)->usize {self.inner.prefill_chunk_min()}
    fn prefill_owns_chunks(&self)->bool {self.inner.prefill_owns_chunks()}
    fn begin_prompt_spare_scope(&mut self)->Result<()> {self.send(call::SPARE_SCOPE,&[],&[]);self.inner.begin_prompt_spare_scope()}
    fn prefill_slice(&mut self,tokens:&[u32],final_chunk:bool,budget:Option<Duration>,prompt_end:usize)->Result<Option<Vec<f32>>> {
        let budget_word=budget.map_or(u64::MAX,|b|u64::try_from(b.as_nanos()).unwrap_or(u64::MAX-1));
        self.send(call::SLICE,&[u64::from(final_chunk),budget_word,prompt_end as u64,u64::from(self.inner.has_draft())],tokens);
        self.inner.prefill_slice(tokens,final_chunk,budget,prompt_end)
    }
    fn can_fork(&self)->bool {self.inner.can_fork()}
    fn can_fork_prefix(&self,position:usize)->bool {self.inner.can_fork_prefix(position)}
    fn keep_prefill_boundary(&self,position:usize)->bool {self.inner.keep_prefill_boundary(position)}
    fn preserves_prefix_boundaries(&self)->bool {self.inner.preserves_prefix_boundaries()}
    fn retain_draft_prefix(&mut self,source:&Self)->Result<bool> {self.send(call::RETAIN_DRAFT,&[source.id],&[]);self.inner.retain_draft_prefix(&source.inner)}
    fn has_draft_prefix(&self)->bool {self.inner.has_draft_prefix()}
    fn fork(&self)->Result<Self> {opened(self.link,self.bypass,call::FORK,&[self.id],||self.inner.fork())}
    fn fork_prefix(&self,position:usize)->Result<Self> {opened(self.link,self.bypass,call::FORK_PREFIX,&[self.id,position as u64],||self.inner.fork_prefix(position))}
    fn retained_bytes(&self)->Option<usize> {self.inner.retained_bytes()}
    fn retained_allocations(&self)->Option<Vec<(usize,usize)>> {self.inner.retained_allocations()}
    fn retention_is_immutable(&self)->bool {self.inner.retention_is_immutable()}
    fn retire(&mut self) {self.send(call::RETIRE,&[],&[]);self.inner.retire()}
    fn retained_private_bytes(&self)->usize {self.inner.retained_private_bytes()}
    fn fork_storage_bytes(&self)->usize {self.inner.fork_storage_bytes()}
    fn settle_prefix(&mut self)->Result<()> {self.send(call::SETTLE,&[],&[]);self.inner.settle_prefix()}
    fn checkpoint_reusable_tokens(&self,tokens:usize)->usize {self.inner.checkpoint_reusable_tokens(tokens)}
    fn retirement_bytes(&self)->Option<usize> {self.inner.retirement_bytes()}
    fn prefix_allocations(&self,tokens:usize)->Option<Vec<(usize,usize)>> {self.inner.prefix_allocations(tokens)}
    fn funded_snapshot_allocations(&self,tokens:Option<usize>)->Vec<(usize,usize)> {self.inner.funded_snapshot_allocations(tokens)}
    fn truncate(&mut self,position:usize)->Result<()> {self.send(call::TRUNCATE,&[position as u64],&[]);Session::truncate(&mut self.inner,position)}
    fn decode_logits(&mut self,rows:usize)->Result<Option<Vec<Vec<f32>>>> {self.send(call::LOGITS,&[rows as u64],&[]);self.inner.decode_logits(rows)}
    fn finish(&mut self)->Result<()> {self.send(call::FINISH,&[],&[]);self.inner.finish()}
}
impl<'e,'a,C:Compute,W:Weights<C>> Mirror<'e,'a,C,W> {
    /// The rows of a batch call on the wire, then the call of half 0 on the chats of its service.
    fn batch<'s>(&'s self,kind:u64,rows:&mut [BatchRows<'_,MirrorSession<'s,'a,C,W>>],run:impl FnOnce(&mut [BatchRows<'_,Chat<'s,'a,C,W>>])->std::result::Result<Vec<Predictions>,BatchError>)->std::result::Result<Vec<Predictions>,BatchError> {
        let mut m=vec![call::BATCH,kind,self.verify_row_cap as u64,rows.len() as u64];
        for row in rows.iter() {
            m.extend_from_slice(&[row.session.id,output_word(row.output),u64::from(row.greedy),row.top_k as u64,u64::from(row.session.inner.has_draft()),row.tokens.len() as u64]);
            m.extend(row.tokens.iter().map(|&t|u64::from(t)));
        }
        self.link.send(m);
        let mut inner=rows.iter_mut().map(|row|BatchRows {session:&mut row.session.inner,tokens:row.tokens,output:row.output,greedy:row.greedy,top_k:row.top_k}).collect::<Vec<_>>();
        run(&mut inner)
    }
}
impl<'e,'a,C:Compute,W:Weights<C>> Engine for Mirror<'e,'a,C,W> {
    type Session<'s>=MirrorSession<'s,'a,C,W> where Self:'s;
    fn session(&self)->Result<Self::Session<'_>> {opened(self.link,self.bypass,call::OPEN,&[],||self.inner.session())}
    fn session_for_admission(&self,request:&Request,tokens:usize)->Result<Self::Session<'_>> {opened(self.link,self.bypass,call::OPEN_FOR,&[tokens as u64],||self.inner.session_for_admission(request,tokens))}
    fn kv_branch_handles(&self)->bool {self.inner.kv_branch_handles()}
    fn active_cohort(&self,chats:usize) {self.link.send(message(call::COHORT,&[chats as u64],&[]));self.inner.active_cohort(chats)}
    fn cohort_cost_control(&self)->bool {false}
    fn shared_batch(&self)->bool {self.inner.shared_batch()}
    fn shared_prefill_batch(&self)->bool {self.inner.shared_prefill_batch()}
    fn greedy_draft_only(&self)->bool {self.inner.greedy_draft_only()}
    fn native_draft_policy(&self)->bool {self.inner.native_draft_policy()}
    fn prefer_shared_prefill(&self,request:&Request)->bool {self.inner.prefer_shared_prefill(request)}
    fn prefix_cache_opt_in(&self)->bool {self.inner.prefix_cache_opt_in()}
    fn permits_prefix_cache(&self,request:&Request)->bool {self.inner.permits_prefix_cache(request)}
    fn shared_prompt_limit(&self)->usize {self.inner.shared_prompt_limit()}
    fn defer_strict_group_draft(&self)->bool {self.inner.defer_strict_group_draft()}
    fn single_chat_draft(&self)->bool {self.inner.single_chat_draft()}
    fn draft_chat_limit(&self)->usize {self.inner.draft_chat_limit()}
    fn discard_contended_draft(&self)->bool {self.inner.discard_contended_draft()}
    fn contended_draft_chats(&self)->usize {self.inner.contended_draft_chats()}
    fn drafts_contended(&self,request:&Request)->bool {self.inner.drafts_contended(request)}
    fn automatic_draft_allowed(&self,chats:usize)->bool {self.inner.automatic_draft_allowed(chats)}
    fn propose_batch<'s>(&'s self,rows:&mut [DraftRows<'_,Self::Session<'s>>])->std::result::Result<Vec<ModelProposal>,BatchError> {
        // Overlap rank 1's deferred target KV/ring accept flush with rank 0's local-only Drafter pass.
        if !rows.is_empty() {
            let mut m=Vec::with_capacity(2+rows.len());
            m.push(call::FLUSH_ACCEPTS);m.push(rows.len() as u64);
            for row in rows.iter() {m.push(row.session.id);}
            self.link.send(m);
        }
        let _guard=BypassGuard::enter(self.bypass);
        let mut inner=rows.iter_mut().map(|row|DraftRows {session:&mut row.session.inner,history:row.history,next:row.next,limit:row.limit}).collect::<Vec<_>>();
        self.inner.propose_batch(&mut inner)
    }
    fn automatic_draft_limit(&self)->usize {self.inner.automatic_draft_limit()}
    fn automatic_draft_min_output(&self)->usize {self.inner.automatic_draft_min_output()}
    fn decode_row_limit(&self,chats:usize)->usize {self.inner.decode_row_limit(chats)}
    fn lone_chat_rows(&self)->usize {self.inner.lone_chat_rows()}
    fn shared_decode_rows(&self)->usize {self.inner.shared_decode_rows()}
    fn shared_decode_min(&self)->usize {self.inner.shared_decode_min()}
    fn tick_row_limit(&self)->Option<usize> {self.inner.tick_row_limit()}
    fn even_cohort_rows(&self)->bool {self.inner.even_cohort_rows()}
    fn shared_prefill_limit(&self)->usize {self.inner.shared_prefill_limit()}
    fn shared_prefill_rows(&self)->usize {self.inner.shared_prefill_rows()}
    fn shared_prefill_max_tokens(&self)->usize {self.inner.shared_prefill_max_tokens()}
    fn shared_prefill_chunk_rows(&self)->usize {self.inner.shared_prefill_chunk_rows()}
    fn whole_prefill_cohort_limits(&self)->(usize,usize) {self.inner.whole_prefill_cohort_limits()}
    fn reserve_foreground_slot(&self)->bool {self.inner.reserve_foreground_slot()}
    fn reserve_full_output(&self)->bool {self.inner.reserve_full_output()}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {self.inner.memory_bytes()}
    fn fatal_error(&self)->Option<String> {self.inner.fatal_error().or_else(||self.link.failed.borrow().clone())}
    fn prefill_batch<'s>(&'s self,rows:&mut [BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<Predictions>,BatchError> {self.batch(PREFILL_BATCH,rows,|rows|self.inner.prefill_batch(rows))}
    fn step_batch<'s>(&'s self,rows:&mut [BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<Predictions>,BatchError> {self.batch(STEP_BATCH,rows,|rows|self.inner.step_batch(rows))}
    fn validate_request(&self,prompt:&[u32],request:&Request)->Result<()> {self.inner.validate_request(prompt,request)}
    fn context_limit(&self)->usize {self.inner.context_limit()}
    fn vocab_size(&self)->usize {self.inner.vocab_size()}
    fn eos_token_ids(&self)->&[u32] {self.inner.eos_token_ids()}
    fn request_reservation_bytes(&self,tokens:usize)->Option<usize> {self.inner.request_reservation_bytes(tokens)}
    fn request_option_bytes(&self,request:&Request)->usize {self.inner.request_option_bytes(request)}
    fn request_reservation_bytes_plain(&self,tokens:usize)->Option<usize> {self.inner.request_reservation_bytes_plain(tokens)}
    fn request_reservation_bytes_plain_cached(&self,tokens:usize,cached:usize)->Option<usize> {self.inner.request_reservation_bytes_plain_cached(tokens,cached)}
    fn request_reservation_bytes_with_retention(&self,tokens:usize,plain:bool,cached:Option<usize>,retain:bool)->Option<usize> {self.inner.request_reservation_bytes_with_retention(tokens,plain,cached,retain)}
    fn request_reservation_bytes_for_prompt(&self,prompt:usize,tokens:usize,plain:bool,cached:Option<usize>,retain:bool)->Option<usize> {self.inner.request_reservation_bytes_for_prompt(prompt,tokens,plain,cached,retain)}
    fn request_reservation_bytes_for_request(&self,prompt:usize,tokens:usize,plain:bool,cached:Option<usize>,retain:bool,request:&Request)->Option<usize> {self.inner.request_reservation_bytes_for_request(prompt,tokens,plain,cached,retain,request)}
    fn session_reservation_bytes(&self)->Option<usize> {self.inner.session_reservation_bytes()}
    fn cache_hold_bytes(&self)->usize {self.inner.cache_hold_bytes()}
    fn unfunded_request_bytes(&self,retain:bool)->usize {self.inner.unfunded_request_bytes(retain)}
    fn cache_free_physical_bytes(&self,live_reserved:usize)->Result<Option<(i128,usize)>> {self.inner.cache_free_physical_bytes(live_reserved)}
    fn cache_room_bytes(&self)->usize {self.inner.cache_room_bytes()}
}

/// Half 1: read the calls of half 0 and make each target call on `service`, until half 0 stops or closes the wire.
/// Half 1 holds no `Drafter` and mirrors target steps, provisional verify passes, and accept/discard transitions.
pub fn follow_service<C:Compute,W:Weights<C>>(service:&Service<'_,C,W>,wire:&mut Wire,template:&Request)->Result<()> {
    let mut chats:HashMap<u64,Chat<'_,'_,C,W>>=HashMap::new();
    let (mut calls,mut failures)=(0u64,0u64);
    while let Some(m)=wire.receive()? {
        let (&call,rest)=m.split_first().ok_or("tp2 mirror: an empty message")?;
        calls+=1;
        if call==call::STOP {break;}
        let outcome=(||->Result<()> {
            let word=|at:usize|rest.get(at).copied().ok_or_else(||format!("tp2 mirror: call {call} is short"));
            let tokens=|from:usize|->Result<Vec<u32>> {rest.get(from..).unwrap_or_default().iter().map(|&t|u32::try_from(t).map_err(|_|"tp2 mirror: a token exceeds its range".to_owned())).collect()};
            let id=word(0)?;
            let chat=|chats:&mut HashMap<u64,Chat<'_,'_,C,W>>|->Result<()> {if chats.contains_key(&id) {Ok(())} else {Err(format!("tp2 mirror: no chat {id}"))}};
            match call {
                call::OPEN=>{chats.insert(id,service.session()?);},
                call::OPEN_FOR=>{chats.insert(id,service.session_for_admission(template,word(1)? as usize)?);},
                call::CLOSE=>{chats.remove(&id);},
                call::FORK=>{let source=chats.get(&word(1)?).ok_or("tp2 mirror: no chat to fork")?;let fork=source.fork()?;chats.insert(id,fork);},
                call::FORK_PREFIX=>{let source=chats.get(&word(1)?).ok_or("tp2 mirror: no chat to fork")?;let fork=source.fork_prefix(word(2)? as usize)?;chats.insert(id,fork);},
                call::COHORT=>service.active_cohort(id as usize),
                call::FLUSH_ACCEPTS=>{
                    let count=id as usize;
                    let mut list=Vec::with_capacity(count);
                    for i in 0..count {
                        if let Some(c)=chats.get(&word(1+i)?) {list.push(c);}
                    }
                    service.follow_flush_accepts(&list)?;
                },
                call::BATCH=>{
                    let (kind,verify_cap,count)=(id,word(1)? as usize,word(2)? as usize);
                    let mut at=3;let mut taken=Vec::with_capacity(count);
                    let parsed=(||->Result<()> {
                        for _ in 0..count {
                            let (chat,output,greedy,_top_k,drafted,len)=(word(at)?,output_of(word(at+1)?)?,word(at+2)?!=0,word(at+3)? as usize,word(at+4)?!=0,word(at+5)? as usize);
                            let row=rest.get(at+6..at+6+len).ok_or("tp2 mirror: a batch row is short")?.iter().map(|&t|u32::try_from(t).map_err(|_|"tp2 mirror: a token exceeds its range".to_owned())).collect::<Result<Vec<_>>>()?;
                            at+=6+len;
                            let session=chats.remove(&chat).ok_or_else(||format!("tp2 mirror: no chat {chat} in a batch"))?;
                            taken.push((chat,session,row,output,greedy,drafted));
                        }
                        Ok(())
                    })();
                    let ran=parsed.and_then(|()|{
                        let mut rows=taken.iter_mut().map(|(_,session,tokens,output,greedy,drafted)|FollowBatchRow {session,tokens,output:*output,greedy:*greedy,drafted:*drafted}).collect::<Vec<_>>();
                        let phase=if kind==PREFILL_BATCH {Phase::Prefill} else {Phase::Decode};
                        service.follow_batch(&mut rows,phase,verify_cap)
                    });
                    for (chat,session,..) in taken {chats.insert(chat,session);}
                    ran?;
                },
                _=>{
                    chat(&mut chats)?;
                    let session=chats.get_mut(&id).ok_or("tp2 mirror: no chat")?;
                    match call {
                        call::STEP=>{let drafted=word(1)?!=0;service.follow_read(session,&tokens(2)?,true,Phase::Decode,drafted)?;},
                        call::ADVANCE=>{let drafted=word(1)?!=0;service.follow_read(session,&tokens(2)?,false,Phase::Prefill,drafted)?;},
                        call::PREFILL=>{let drafted=word(1)?!=0;service.follow_read(session,&tokens(2)?,true,Phase::Prefill,drafted)?;},
                        call::PREFILL_ADVANCE=>{let drafted=word(1)?!=0;service.follow_read(session,&tokens(2)?,false,Phase::Prefill,drafted)?;},
                        call::SLICE=>{
                            let (final_chunk,drafted)=(word(1)?!=0,word(4)?!=0);
                            service.follow_read(session,&tokens(5)?,final_chunk,Phase::Prefill,drafted)?;
                        },
                        call::ACCEPT_DRAFT=>{
                            service.follow_accept(session,word(1)? as usize,word(2)? as usize,word(3)?!=0)?;
                        },
                        call::SETTLE=>session.settle_prefix()?,
                        call::ADMISSION=>session.admission_tokens(word(1)? as usize),
                        call::RESERVE_DRAFT=>session.reserve_model_draft(word(1)? as usize),
                        call::NO_DRAFT=>session.disable_model_draft(),
                        call::DRAFT_OVER=>{session.draft_over();},
                        call::CONFIGURE=>session.configure_request(template)?,
                        call::RETENTION=>session.set_cache_retention(word(1)?!=0),
                        call::RELEASE=>session.release_replay_snapshot(),
                        call::AUTOMATIC=>session.automatic_draft(word(1)? as usize,word(2)? as usize),
                        call::SPARE_SCOPE=>session.begin_prompt_spare_scope()?,
                        call::RETIRE=>session.retire(),
                        call::TRUNCATE=>Session::truncate(session,word(1)? as usize)?,
                        call::LOGITS=>{session.decode_logits(word(1)? as usize)?;},
                        call::FINISH=>session.finish()?,
                        call::RETAIN_DRAFT=>{
                            let mut session=chats.remove(&id).ok_or("tp2 mirror: no chat")?;
                            let kept=match chats.get(&word(1)?) {Some(source)=>session.retain_draft_prefix(source).map(|_|()),None=>Err("tp2 mirror: no source chat".to_owned())};
                            chats.insert(id,session);kept?;
                        },
                        _=>return Err(format!("tp2 mirror: no call {call}")),
                    }
                },
            }
            Ok(())
        })();
        // Half 0 made the same call on the same state: a call that fails there fails here, and the service goes on.
        if let Err(error)=outcome {failures+=1;eprintln!("TP2 FOLLOW call={call} error={error}");}
    }
    eprintln!("TP2 FOLLOW END calls={calls} failures={failures} open_chats={}",chats.len());
    Ok(())
}
