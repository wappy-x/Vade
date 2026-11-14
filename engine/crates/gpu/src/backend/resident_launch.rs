//! Queue-retained operands and a stable indirect table permit larger graph pieces.
use super::*;
use crate::{resident_launch::{Command,Kind},resident_launch_encode as encode};
use std::collections::BTreeMap;
#[derive(Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
struct Grid {kind:Kind,blocks:u32,rows:u32,planes:u32,threads:u32,shared:u32,lane:u8}
/// Lanes of a recorded call: MAIN runs on the capture stream; SIDE runs on a stream forked from it at the call's
/// place, after every earlier main call; JOIN is a main call that first waits for all side calls before it.
const MAIN:u8=0;const SIDE:u8=1;const JOIN:u8=2;
struct Call {grid:Grid,command:Command,copy_bytes:u64}
#[path="resident_launch_shadow.rs"] mod shadow;
// Recorder declares the executables before their shared table, so they drop first.
struct Piece<'a> {graph:DecodeGraph<'a>,used:u64}
/// After a real wait the GPU is idle. The recorder then submits a short first
/// part and longer later parts, so the GPU works while the host records. The
/// limits count arithmetic commands and depend only on the recorded order.
/// `fed` is the number of arithmetic commands submitted since that wait. A part may
/// hold as many as all the parts before it (1, 1, 2, 4, 8, ...): when it arrives the
/// device still has about half of the fed work to do. A part that an upload forces out
/// early, with one or two commands, moves the ramp by those commands only. (When the
/// ramp counted parts, four such parts after a selection read ended it, and the next
/// 261 commands, three layers, waited in the recorder while the device was empty.)
/// After RAMP commands the host is far ahead and a part has no limit.
/// This ramp runs only after a selection read (`tail`). After any other wait the ramp counts parts,
/// as at first (1, 4, 16, 64 commands, then no limit), but for the weight batch that `leaves_alone`
/// names below. Reason: a put's doorbell is queued when its part is submitted. With the doubling ramp at
/// every step start (four more parts) the short cell of the CPU-chain train ran 1.5 percent slower,
/// in runs of slow steps on both ranks (train-5a, 2026-11-09).
const RAMP:usize=128;
fn part_limit(fed:usize,tail:bool)->usize {
    if tail {match fed {0=>1,n if n<RAMP=>1<<n.ilog2(),_=>usize::MAX}} else {match fed {0=>1,1=>4,2=>16,3=>64,_=>usize::MAX}}
}
/// The second part after a wait that is not a selection read leaves alone when its first arithmetic command is a weight
/// batch, and it does not move the ramp. Reason: since the boundary is one fused launch, a decode step's first part is the
/// boundary's own short chain (about 20 us of device work) and the second part starts with the step's first weight batch
/// (about 250 us). With four commands in that part the device waited for the host to record three more: 170 us a step
/// with nothing queued (254 traced steps of one chat, 2026-11-12). Only a batch leaves alone: a part that starts with any
/// other command keeps the limit of four, so a causal chain whose first member arrives there still fuses
/// (`chain_counts_once_in_the_ramp` of the check tool). Recording and the flush both ask this function.
/// Because the ramp does not move, the rule holds for every part while `fed` is 1: weight batches that follow each other
/// there leave one by one (one more launch each; the code sets no cap), and the first part that starts with any other
/// command takes the limit of four and moves the ramp on.
fn leaves_alone(fed:usize,tail:bool,first:Option<Kind>)->bool {!tail && fed==1 && matches!(first,Some(Kind::MixedBf16|Kind::Mixed|Kind::PlainBatch))}
fn first_arithmetic(calls:&[Call])->Option<Kind> {calls.iter().map(|call|call.grid.kind).find(|&kind|arithmetic(kind))}
/// The ramp's count. `append_resident_bytes` counts a command as it is recorded and `flush_resident_launches` counts the
/// recorded calls again to cut the parts. Both must give the same number for the same calls, or the part that recording
/// asks for is never a part that a flush captured. A fused causal chain is one command: its three members count as one
/// from the moment they fuse.
fn arithmetic(kind:Kind)->bool {
    matches!(kind,Kind::AbsorbDense|Kind::AbsorbAttention
        |Kind::Causal|Kind::Expert|Kind::Plain|Kind::PlainBf16|Kind::Mx|Kind::Mixed|Kind::MixedBf16|Kind::PlainBatch|Kind::NvStream|Kind::CausalFast|Kind::CausalChain
        |Kind::NvTile|Kind::DeviceRoute|Kind::NvQuarters)
}
fn zero_grid(zeros:&[(usize,u64,u32,bool)])->Option<Grid> {
    let longest=zeros.iter().map(|z|z.2).max()?;Some(Grid {kind:Kind::Copy,blocks:longest.div_ceil(128),rows:zeros.len() as u32,planes:1,threads:128,shared:0,lane:MAIN})
}
/// Operand tables of recorded commands share one private device area. One upload
/// per piece carries them, so a join or a batch does not end the piece.
use crate::resident_launch::stage::{BYTES as STAGE_BYTES,BASE as ZERO_BYTES,full as stage_full};
// Commands end where the zero rows start. Their start depends only on the
// piece's node count, so each graph has a stable address without upload padding.
const COMMAND_BYTES:usize=crate::GraphAllowance::FLASH_MAX_NODES*core::mem::size_of::<Command>();
const STAGE_BASE:usize=COMMAND_BYTES+ZERO_BYTES;
/// Zero fills of fresh private outputs share one copy kernel at the start of a
/// piece. Its rows read one zero source. The row table starts the stage area.
const ZERO_ROWS:usize=crate::resident_launch::stage::ZERO_ROWS;const ZERO_WORDS:usize=1<<15;
#[derive(Default)]
pub(super) struct Recorder<'a> {enabled:bool,paused:bool,pub(super) failure:Option<String>,calls:Vec<Call>,pieces:BTreeMap<Vec<Grid>,Piece<'a>>,
    stage:Option<DeviceVector<'a>>,staged:Vec<u8>,bells:Vec<(usize,std::sync::Arc<crate::tp_mailbox_host::Bell>,u64)>,
    zero_source:Option<DeviceVector<'a>>,zeros:Vec<(usize,u64,u32,bool)>,fresh:Vec<(u64,u64)>,fresh_at:u64,recorded:u64,
    // Fresh fills not yet placed (index, target, words); see `fuse_fresh_fills`. A proof run's poison words.
    waiting:Vec<(usize,u64,u32)>,poison_source:Option<DeviceVector<'a>>,
    heavy:usize,fed:usize,tail:bool,clock:u64,copies:crate::resident_launch::CopyCounts,seen_keys:crate::resident_launch::capture_keys::Seen,
    // SHADOW-STREAM: the second stream a side-branch prefetch is captured on, and whether prefetch-only launches use it.
    side:Option<Stream<'a>>,side_off:bool,side_placed:u64,
    // Print only. `ramp`: parts that left when recording reached their limit (the executable existed); limits reached with no
    // executable; parts that then left late, with a later flush. `missed`, `early`: this part's state for the third count.
    // `shadow`: side prefetches not placed because a BF16 batch launch lies between the take and the target, because the
    // piece had no room, or because the take had one already.
    ramp:[u64;3],missed:bool,early:bool,shadow:[u64;3],
    // PROJ-DELIVERY side lane: recording into it, side calls not yet joined, a join asked for the next main call; its stream and
    // fork/join events (a stream of its own, so the SHADOW-STREAM branch keeps its edges).
    lane_side:bool,lane_open:bool,lane_join:bool,lane_fork:Option<(Stream<'a>,Event<'a>,Event<'a>)>,
    // `drained_at`: the piece clock at the last real stream wait. A piece used at or before it has no replay in flight.
    // `updates` (print only): pieces whose retained executable took new launch grids in place, with no capture of a new one.
    drained_at:u64,updates:u64}
impl Recorder<'_> {
    /// The ramp starts again (`Backend::feed_resident_launches`: the caller says the device is fed; no stream wait proves it).
    fn fed(&mut self) {self.fed=0;self.tail=false;}
    /// A real stream wait has ended: nothing is queued on the device, so no retained executable has a replay in flight.
    pub(super) fn drained(&mut self) {self.fed();self.drained_at=self.clock;}
}
/// Retained executables of one kernel sequence before one of them is reused for a new grid set. Below it a new key gets
/// its own executable, so keys that come back in a short cycle (the four phases of a pooled step, the four row counts past
/// the selection budget) stay hits. At it, a grid set that is new on every step (an extent that follows the context) takes
/// the executable used longest ago, and the set stops growing.
const SAME_KERNELS:usize=8;
/// True when two pieces launch the same kernels in the same order with the same block sizes on the same lanes: only launch
/// grids differ. The executable of one can then be updated in place to run the other (`Backend::end_segment`). The lane
/// is part of it: a SIDE or JOIN call adds fork and join edges to the captured graph, so a piece whose call moved to the
/// side stream is another graph, not the same one with other grids.
fn same_kernels(a:&[Grid],b:&[Grid])->bool {a.len()==b.len() && a.iter().zip(b).all(|(a,b)|a.kind==b.kind && a.threads==b.threads && a.shared==b.shared && a.lane==b.lane)}
// Backend::side_launch: during a piece capture, the forked side stream that takes the next launch (null: the capture stream).
impl<'a> Backend<'a> {
    /// Only the owned GLM queue enables recording. Ordinary callers stay eager.
    pub fn enable_resident_launches(&self)->Result<(),String> {
        self.require_resident_stream()?;
        if !self.resident_queue_active() {return Err("resident launch recording requires an owned queue".into());}
        self.require_arm_kernels(&["resident_launch_causal","resident_launch_copy_words","resident_launch_zero_words"])?;
        self.recorder.borrow_mut().enabled=true;Ok(())
    }
    /// Submit what is recorded and restart the feed count, as a real wait does, with no
    /// wait. Pieces keep the boundaries they had when the host read a value at this point.
    pub fn feed_resident_launches(&self)->Result<(),String> {self.flush_resident_launches()?;self.recorder.borrow_mut().fed();Ok(())}
    /// The wait that just ended was a selection read: the rest of the layer is recorded while the device is empty, and uploads
    /// force small parts out first. Until the next real wait the ramp counts commands (see `part_limit`). Part boundaries only.
    pub fn resident_selection_read(&self) {if let Ok(mut recorder)=self.recorder.try_borrow_mut() {recorder.tail=true;}}
    /// Proof and bench switch: false keeps the take's prefetch in the take's own tail rows (the form before SHADOW-STREAM).
    pub fn set_shadow_side(&self,on:bool) {self.recorder.borrow_mut().side_off = !on;}
    /// Recording remains enabled while a prompt row pass pauses it.
    pub fn resident_launches_enabled(&self)->bool {self.recorder.borrow().enabled}
    pub fn resident_launches_active(&self)->bool {let r=self.recorder.borrow();r.enabled && !r.paused}
    pub(super) fn disable_resident_launches(&self)->Result<(),String> {let result=self.flush_resident_launches();self.recorder.borrow_mut().enabled=false;result}
    pub fn resident_copy_counts(&self)->crate::resident_launch::CopyCounts {self.recorder.borrow().copies}
    /// Print only: pieces whose retained executable was updated in place for new launch grids (no new capture).
    pub fn resident_piece_updates(&self)->u64 {self.recorder.borrow().updates}
    /// Print only: ramp parts that left early, part limits reached with no executable, parts that left late; then side
    /// prefetches not placed across a BF16 batch launch, for lack of room in the piece, and at a take that had one.
    pub fn resident_ramp_counts(&self)->[u64;6] {let r=self.recorder.borrow();[r.ramp[0],r.ramp[1],r.ramp[2],r.shadow[0],r.shadow[1],r.shadow[2]]}
    /// A recorded put wakes the wire only after its part reaches the stream.
    /// Retain the bell through submission, as the queue retains its GPU operands.
    pub(crate) fn resident_mailbox_queued(&self,bell:&std::sync::Arc<crate::tp_mailbox_host::Bell>,sequence:u64) {
        let mut r=self.recorder.borrow_mut();
        if r.enabled && !r.paused && r.calls.last().is_some_and(|call|matches!(call.grid.kind,Kind::Put|Kind::PutWide|Kind::PutPlain|Kind::PutJoin)) {
            let at=r.calls.len()-1;r.bells.push((at,bell.clone(),sequence));
        } else {bell.queued(sequence);}
    }
    pub(super) fn prefetch_route_window(&self,build:crate::device_route::Build,bytes:usize) {
        if build.down==0 || !self.resident_launches_active() || build.picks==0 {return;}
        let window=crate::bus_shadow::Window {weight:build.table,bytes:(bytes as u64).min(crate::bus_shadow::BUDGET/u64::from(build.picks)),ids:build.ids,experts:build.experts,picks:build.picks};
        shadow::routed_target(&mut self.recorder.borrow_mut(),window);
        let placed=std::mem::take(&mut self.recorder.borrow_mut().side_placed);if placed!=0 {self.stats.borrow_mut().recorded_kinds[Kind::ShadowSide as usize]+=u128::from(placed);}
    }
    /// Calls recorded by `run` go to the side lane: in the captured piece they run on a stream forked at their place,
    /// so they may overlap the main calls recorded after them, until `resident_side_join`. The caller guarantees that no
    /// main call recorded before that join reads or writes what these calls write, or writes what they read.
    /// With recording off the calls run eagerly in call order, as before.
    pub fn resident_side<T>(&self,run:impl FnOnce()->Result<T,String>)->Result<T,String> {
        if !self.resident_launches_active() {return run();}
        if self.recorder.borrow().lane_fork.is_none() {
            let fork=(Stream::new(self.context)?,Event::ready(&self.stream)?,Event::ready(&self.stream)?);self.recorder.borrow_mut().lane_fork=Some(fork);
        }
        self.recorder.borrow_mut().lane_side=true;let result=run();self.recorder.borrow_mut().lane_side=false;result
    }
    /// The next main call waits for every side call recorded before it.
    pub fn resident_side_join(&self) {let mut r=self.recorder.borrow_mut();if r.lane_open {r.lane_join=true;}}
    fn append_resident(&self,kind:Kind,command:Command,grid:[u32;5])->Result<(),String> {
        let words=match kind {Kind::CopyWords|Kind::ZeroWords=>u64::from_ne_bytes(command.payload[16..24].try_into().unwrap()),
            Kind::Copy=>u32::from_ne_bytes(command.payload[16..20].try_into().unwrap()) as u64,_=>0};
        self.append_resident_bytes(kind,command,grid,words.saturating_mul(4))
    }
    fn append_resident_bytes(&self,kind:Kind,command:Command,grid:[u32;5],copy_bytes:u64)->Result<(),String> {
        self.settle_fresh_fills();
        if grid[0]==0 {return Ok(());}
        if grid[1]==0 || grid[1]>65535 || grid[2]==0 || grid[2]>65535 || grid[3]==0 || grid[3]>1024 || !grid[3].is_multiple_of(32) {return Err("resident recorded grid exceeds driver limits".into());}
        if self.recorder.borrow().calls.len()>=crate::GraphAllowance::FLASH_MAX_NODES-1 {self.flush_resident_launches()?;}
        self.context.resident_unsubmitted.set(true);self.context.pending.set(true);self.stats.borrow_mut().recorded_kinds[kind as usize]+=1;
        let mut r=self.recorder.borrow_mut();r.recorded=r.recorded.wrapping_add(1);
        shadow::static_target(&mut r,&command);
        let placed=std::mem::take(&mut r.side_placed);if placed!=0 {self.stats.borrow_mut().recorded_kinds[Kind::ShadowSide as usize]+=u128::from(placed);}
        // PROJ-DELIVERY lanes. A call fuses into earlier calls only when they are on its lane (a MAIN call may also fuse into a
        // JOIN call, which keeps its wait first); a JOIN call never fuses, so its wait stays before it.
        let lane=if r.lane_side {r.lane_open=true;SIDE} else if std::mem::take(&mut r.lane_join) && std::mem::take(&mut r.lane_open) {JOIN} else {MAIN};
        let fusable=|calls:&[Call],n:usize|->bool {lane!=JOIN && calls.len()>=n && calls[calls.len()-n..].iter().all(|c|c.grid.lane==lane || lane==MAIN && c.grid.lane==JOIN)};
        let fused_lane=|calls:&[Call],n:usize|->u8 {if calls[calls.len()-n..].iter().any(|c|c.grid.lane==JOIN) {JOIN} else {lane}};
        if kind==Kind::Round {
            if let Some(previous)=r.calls.last_mut().filter(|call|call.grid.kind==Kind::NormTail && (call.grid.lane==lane || lane==MAIN && call.grid.lane==JOIN) && lane!=JOIN) {
                let output=&previous.command.payload[24..32];let width=&previous.command.payload[32..36];
                if output==&command.payload[..8] && width==&command.payload[16..20] && previous.command.payload[144..148]==[0;4] {
                    previous.command.put(128,&command.payload[..20]);return Ok(());
                }
            }
        }
        let count=r.calls.len();let chain=if count>=2 && fusable(&r.calls,2) {crate::resident_causal_chain::fuse(&r.calls[count-2].command,&r.calls[count-1].command,&command)} else {None};
        if let Some((command,blocks))=chain {
            // The two members recorded before counted one each. The chain that replaces all three counts once, below.
            let start=count-2;let chained=fused_lane(&r.calls,2);r.calls.truncate(start);r.heavy=r.heavy.saturating_sub(2);
            for zero in &mut r.zeros {if zero.0>start {zero.0=zero.0.saturating_sub(2).max(start);}}
            r.calls.push(Call {grid:Grid {kind:Kind::CausalChain,blocks,rows:1,planes:1,threads:128,shared:0,lane:chained},command,copy_bytes:0});
        } else {
        if let Some(previous)=r.calls.last().filter(|_|fusable(&r.calls,1)) {
            let stage=r.stage.as_ref().map_or(0,|value|value.device_address()+STAGE_BASE as u64);
            if let (Some(first),Some(next))=(crate::resident_pointwise::Plan::read(&previous.command,&r.staged,stage,previous.grid.rows),
                crate::resident_pointwise::Plan::read(&command,&r.staged,stage,grid[1])) {
                if let Some((command,blocks))=first.then(next) {
                    let merged=fused_lane(&r.calls,1);let previous=r.calls.last_mut().unwrap();previous.command=command;
                    previous.grid=Grid {kind:Kind::Pointwise,blocks,rows:1,planes:1,threads:128,shared:0,lane:merged};previous.copy_bytes=previous.copy_bytes.saturating_add(copy_bytes);
                    return Ok(());
                }
            }
        }
        r.calls.push(Call {grid:Grid {kind,blocks:grid[0],rows:grid[1],planes:grid[2],threads:grid[3],shared:grid[4],lane},command,copy_bytes});
        }
        if arithmetic(kind) {r.heavy+=1;
            // Submit this part now only when its executable exists: a capture
            // must first drain the stream, and that would stop the device.
            // Never while a side call waits for its join: the flush keeps them in one part.
            let limit=if leaves_alone(r.fed,r.tail,first_arithmetic(&r.calls)) {1} else {part_limit(r.fed,r.tail)};
            if r.heavy==limit && !r.lane_open {let key=zero_grid(&r.zeros).into_iter().chain(r.calls.iter().map(|c|c.grid)).collect::<Vec<_>>();
                if r.pieces.contains_key(&key) {r.ramp[0]+=1;r.early=true;drop(r);return self.flush_resident_launches();}
                r.ramp[1]+=1;r.missed=true;}
        }Ok(())
    }
    /// A new allocation is fresh until the next recorded command. No pending
    /// command names it: an owned queue keeps every freed allocation out of the
    /// pool until its real drain.
    pub(super) fn note_fresh_allocation(&self,address:u64,bytes:usize) {
        let Ok(mut r)=self.recorder.try_borrow_mut() else {return;};
        if !r.enabled || r.paused {return;}
        if r.fresh_at!=r.recorded || r.fresh.len()>=16 {r.fresh.clear();r.fresh_at=r.recorded;}
        r.fresh.push((address,bytes as u64));
    }
    /// Fresh fills that no copy took run as the piece's merged fill, at the index where they were made: the head placement.
    /// No command was recorded since they were made (any record settles them first), so that index is still the next one.
    fn settle_fresh_fills(&self) {
        let mut r=self.recorder.borrow_mut();if r.waiting.is_empty() {return;}
        let waiting=std::mem::take(&mut r.waiting);r.zeros.extend(waiting.into_iter().map(|(at,target,words)|(at,target,words,false)));
    }
    /// A copy whose rows land inside waiting fresh outputs takes, in its own launch, the zero rows for the words of those
    /// outputs that it does not write, so no zero line is written long before its producer (DEAD-ZERO).
    /// Proof that the bytes equal the head's fill-then-copy on every path: (1) no command was recorded between the output's
    /// allocation and this copy (fresh, then waiting until the next record), so nothing reads or writes those words in
    /// between, and the copy is the first command to touch them; (2) the copy kernel (flash_hybrid_copy) stores word i of
    /// each row for every i < words with one raw store (grid x covers the longest row, grid y one row each); (3) the zero
    /// rows cover exactly the rest of each output, disjoint from every copy row's target and source. After this launch each
    /// output holds the head's bytes: copied words where the copy writes, zero elsewhere, including the words a later take
    /// may fail to write. A row that crosses an output's edge, or reads a waiting output, keeps the head placement.
    /// A proof run in poison mode also fills the copied words with POISON first, so a word the copy misses shows.
    fn fuse_fresh_fills(&self,rows:&[(u64,u64,u32)])->Result<bool,String> {
        let mode=crate::fill_free::mode();
        if mode==crate::fill_free::Mode::Zero || rows.is_empty() {return Ok(false);}
        let (waiting,zero)={let r=self.recorder.borrow();(r.waiting.clone(),r.zero_source.as_ref().map(|v|v.device_address()))};
        let Some(zero)=zero else {return Ok(false);};
        let span=|target:u64,words:u32|(target,target+words as u64*4);
        let mut fused=vec![false;waiting.len()];
        for &(source,target,words) in rows {
            let ((a,b),(sa,sb))=(span(target,words),span(source,words));let mut inside=None;
            for (i,&(_,t,w)) in waiting.iter().enumerate() {let (s,e)=span(t,w);
                if sa<e && s<sb {return Ok(false);}
                if a>=s && b<=e {inside=Some(i);} else if a<e && s<b {return Ok(false);}
            }
            if let Some(i)=inside {fused[i]=true;}
        }
        if !fused.contains(&true) {return Ok(false);}
        let chunk=rows.iter().map(|r|r.2).max().unwrap_or(0).clamp(128,ZERO_WORDS as u32) as u64;
        let (mut table,mut poison)=(Vec::new(),Vec::new());
        for &(source,target,words) in rows {table.push((source,target,words));}
        for (i,&(_,t,w)) in waiting.iter().enumerate() {
            if !fused[i] {continue;}
            let (s,e)=span(t,w);
            let mut covered=rows.iter().map(|&(_,t,w)|span(t,w)).filter(|&(a,b)|a>=s && b<=e).collect::<Vec<_>>();covered.sort_unstable();
            let mut cursor=s;let mut gaps=Vec::new();
            for &(a,b) in &covered {if a>cursor {gaps.push((cursor,a));}cursor=cursor.max(b);}
            if cursor<e {gaps.push((cursor,e));}
            for (mut a,b) in gaps {while a<b {let n=((b-a)/4).min(chunk);table.push((zero,a,n as u32));a+=n*4;}}
            if mode==crate::fill_free::Mode::Poison {
                for (mut a,b) in covered {while a<b {let n=((b-a)/4).min(ZERO_WORDS as u64);poison.push((a,n as u32));a+=n*4;}}
            }
        }
        let longest=table.iter().map(|r|r.2).max().unwrap_or(0);
        if table.len()>u16::MAX as usize || table.len()*24>STAGE_BYTES/4 || poison.len()>ZERO_ROWS {return Ok(false);}
        self.require_arm_kernels(&["flash_hybrid_copy"])?;
        {   // The other waiting fills keep the head placement, before this copy.
            let mut r=self.recorder.borrow_mut();r.waiting.clear();
            r.zeros.extend(waiting.iter().zip(&fused).filter(|(_,f)|!**f).map(|(&(at,target,words),_)|(at,target,words,false)));
        }
        if !poison.is_empty() {
            if self.recorder.borrow().zeros.len()+poison.len()>ZERO_ROWS {self.flush_resident_launches()?;}
            let mut r=self.recorder.borrow_mut();let at=r.calls.len();r.zeros.extend(poison.into_iter().map(|(target,words)|(at,target,words,true)));
        }
        let mut bytes=Vec::with_capacity(table.len()*24);
        for (source,target,words) in &table {bytes.extend_from_slice(&source.to_ne_bytes());bytes.extend_from_slice(&target.to_ne_bytes());
            bytes.extend_from_slice(&words.to_ne_bytes());bytes.extend_from_slice(&0u32.to_ne_bytes());}
        let grid=[longest.div_ceil(128),table.len() as u32,1,128,0];
        if !self.record_resident_tabled(Kind::Copy,grid,&bytes,|address|encode::copy(crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:address}))? {
            return Err("a fused fresh-output copy was not recorded".into());
        }
        // Counted as zero fill rows and words in no launch of their own.
        self.note_zero_fill(0,(table.len()-rows.len()) as u128,table[rows.len()..].iter().map(|r|r.2 as usize).sum());
        Ok(true)
    }
    /// Called only with the original checked kernel ABI at the GPU boundary.
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    pub(super) unsafe fn record_resident_launch(&self,name:&str,grid:[u32;5],args:&[*mut c_void])->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        if !self.resident_launches_active() {return Ok(false);}
        // SAFETY: launch_flash_grid's caller supplies this registered name's
        // original ABI pointers. These typed reads copy live argument values;
        // serializers copy numeric fields separately, never Rust padding. No
        // borrowed host pointer enters the retained command. Queue-owned GPU
        // operands survive all deferred submission and completion boundaries.
        let command=unsafe {match name {
            "latent_absorption"|"latent_absorption_scores"|"latent_absorption_values"=>{
                let kind=if name=="latent_absorption" {Kind::AbsorbDense} else {Kind::AbsorbAttention};
                let mut c=encode::causal(args[0].cast::<crate::causal_ops::Arguments>().read_unaligned());c.kind=kind as u32;Some((kind,c))},
            "causal_operation"=>Some((Kind::Causal,encode::causal(args[0].cast::<crate::causal_ops::Arguments>().read_unaligned()))),
            "expert_operation"=>Some((Kind::Expert,encode::expert(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned()))),
            "causal_fast"=>Some((Kind::CausalFast,encode::causal_fast(args[0].cast::<crate::causal_ops::Arguments>().read_unaligned()))),
            "nv_rows_stream"=>Some((Kind::NvStream,encode::nv_stream(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned()))),
            "nv_tile"|"nv_cols"=>Some((Kind::NvTile,encode::nv_tile(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned()))),
            // The pair split: raw quarter accumulators, the fold inside each quarter, the take of the sum exchange.
            "nv_cols_q4"|"nv_cols_q2"|"nv_cols_q1"=>Some((Kind::NvQuarters,encode::nv_quarters(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned(),
                match name {"nv_cols_q4"=>4,"nv_cols_q2"=>2,_=>1}))),
            "nv_quarter_fold"=>Some((Kind::QuarterFold,encode::quarter_fold(args[0].cast::<crate::pair_split::QuarterFold>().read_unaligned()))),
            "tp_reduce_take"=>{if args.len()!=9 {return Err("resident REDUCE argument count differs".into());}
                let mut c=Command::new(Kind::ReduceTake);for (i,p) in args.iter().enumerate() {c.word(i,p.cast::<u64>().read_unaligned());}Some((Kind::ReduceTake,c))},
            "device_route_build"=>Some((Kind::DeviceRoute,encode::device_route(args[0].cast::<crate::device_route::Build>().read_unaligned()))),
            "flash_next_mxfp8_m1"=>Some((Kind::Mx,encode::mx(args[0].cast::<crate::next_params::MxLayer>().read_unaligned()))),
            "flash_resident_projection_batch"=>Some((Kind::Mixed,encode::mixed(args[0].cast::<crate::next_params::ResidentProjectionBatch>().read_unaligned(),Kind::Mixed))),
            "flash_resident_projection_bf16"=>Some((Kind::MixedBf16,encode::mixed(args[0].cast::<crate::next_params::ResidentProjectionBatch>().read_unaligned(),Kind::MixedBf16))),
            "flash_resident_projection_plain_bf16"=>Some((Kind::PlainBatch,encode::mixed(args[0].cast::<crate::next_params::ResidentProjectionBatch>().read_unaligned(),Kind::PlainBatch))),
            "bf16_lanes_round"=>Some((Kind::Round,encode::round(args[0].cast::<crate::mhc_boundary::LaneCast>().read_unaligned()))),
            "flash_hybrid_copy"=>Some((Kind::Copy,encode::copy(args[0].cast::<crate::kernel_params::HybridCopy>().read_unaligned()))),
            "mhc_prepare"=>Some((Kind::Prepare,encode::prepare(args[0].cast::<crate::mhc_params::Prepare>().read_unaligned()))),
            "mhc_bf16_collapse"=>Some((Kind::Collapse,encode::collapse(args[0].cast::<crate::mhc_params::Collapse>().read_unaligned()))),
            "mhc_bf16_combine"=>Some((Kind::Combine,encode::combine(args[0].cast::<crate::mhc_params::Combine>().read_unaligned()))),
            "mhc_boundary_rms"=>Some((Kind::MhcBoundaryRms,encode::boundary(args[0].cast::<crate::mhc_params::Boundary>().read_unaligned(),Kind::MhcBoundaryRms))),
            "mhc_boundary_dots"=>Some((Kind::MhcBoundaryDots,encode::boundary(args[0].cast::<crate::mhc_params::Boundary>().read_unaligned(),Kind::MhcBoundaryDots))),
            "mhc_boundary_finish"=>Some((Kind::MhcBoundaryFinish,encode::boundary(args[0].cast::<crate::mhc_params::Boundary>().read_unaligned(),Kind::MhcBoundaryFinish))),
            "mhc_boundary_fused"=>Some((Kind::MhcBoundaryFused,encode::fused(args[0].cast::<crate::mhc_params::FusedBoundary>().read_unaligned()))),
            "model_rms_partials"=>Some((Kind::RmsPartials,encode::rms(args[0].cast::<crate::kernel_params::ModelRms>().read_unaligned(),Kind::RmsPartials))),
            "model_rms_apply"=>Some((Kind::RmsApply,encode::rms(args[0].cast::<crate::kernel_params::ModelRms>().read_unaligned(),Kind::RmsApply))),
            "model_elementwise"=>Some((Kind::Elementwise,encode::elementwise(args[0].cast::<crate::kernel_params::ModelElementwise>().read_unaligned()))),
            "project_f32_ordered"=>{if args.len()!=5 {return Err("ordered projection argument count differs".into());}let mut c=Command::new(Kind::Plain);for i in 0..3 {c.word(i,args[i].cast::<u64>().read_unaligned());}for i in 3..5 {c.word(i,args[i].cast::<u32>().read_unaligned() as u64);}Some((Kind::Plain,c))},
            "project_bf16_ordered"=>{if args.len()!=5 {return Err("ordered BF16 projection argument count differs".into());}let mut c=Command::new(Kind::PlainBf16);for i in 0..3 {c.word(i,args[i].cast::<u64>().read_unaligned());}for i in 3..5 {c.word(i,args[i].cast::<u32>().read_unaligned() as u64);}Some((Kind::PlainBf16,c))},
            "tp_mailbox_put_wide"=>{if args.len()!=8 {return Err("resident mailbox argument count differs".into());}let mut c=Command::new(Kind::PutWide);for (i,p) in args.iter().enumerate() {c.word(i,p.cast::<u64>().read_unaligned());}Some((Kind::PutWide,c))},
            // The mHC host lane: its put (no payload check: a nonfinite block must reach the same later checks as before) and its prefetch-only launch.
            "tp_mailbox_put"=>{if args.len()!=9 {return Err("resident mailbox argument count differs".into());}let mut c=Command::new(Kind::PutPlain);for (i,p) in args.iter().enumerate() {c.word(i,p.cast::<u64>().read_unaligned());}Some((Kind::PutPlain,c))},
            "chain_shadow_prefetch"=>{if args.len()!=1 {return Err("resident prefetch argument count differs".into());}let p=args[0].cast::<crate::mhc_params::ShadowPrefetch>().read_unaligned();
                let mut c=Command::new(Kind::ShadowPrefetch);c.word(0,p.prefetch);c.word(1,p.prefetch_bytes);c.put(16,&p.prefetch_policy.to_ne_bytes());Some((Kind::ShadowPrefetch,c))},
            "tp_mailbox_put_finite"|"tp_mailbox_take"=>{if args.len()!=10 {return Err("resident mailbox argument count differs".into());}let kind=if name=="tp_mailbox_take" {Kind::Take} else {Kind::Put};let mut c=Command::new(kind);for (i,p) in args.iter().enumerate() {c.word(i,p.cast::<u64>().read_unaligned());}Some((kind,c))},
            _=>None,
        }};
        if let Some((kind,command))=command {self.append_resident(kind,command,grid)?;Ok(true)} else {self.flush_resident_launches()?;Ok(false)}
    }
    /// Record one command together with the small table it reads. The table is
    /// uploaded with this command's piece and is valid only for that replay, so
    /// both always enter the same piece. False: the caller uses its own table.
    pub(super) fn record_resident_tabled(&self,kind:Kind,grid:[u32;5],table:&[u8],build:impl FnOnce(u64)->Command)->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        let bytes=table.len().div_ceil(16)*16;
        if !self.resident_launches_active() || grid[0]==0 || bytes==0 || bytes>STAGE_BYTES/4 {return Ok(false);}
        if kind==Kind::Copy && table.len().is_multiple_of(24) && !self.recorder.borrow().waiting.is_empty() {
            let rows=table.as_chunks::<24>().0.iter().map(|row|(u64::from_ne_bytes(row[0..8].try_into().unwrap()),u64::from_ne_bytes(row[8..16].try_into().unwrap()),u32::from_ne_bytes(row[16..20].try_into().unwrap()))).collect::<Vec<_>>();
            if self.fuse_fresh_fills(&rows)? {return Ok(true);}
        }
        if self.recorder.borrow().stage.is_none() {let stage=self.vector((COMMAND_BYTES+STAGE_BYTES)/4)?;self.recorder.borrow_mut().stage=Some(stage);}
        let (nodes,full)={let r=self.recorder.borrow();(r.calls.len()>=crate::GraphAllowance::FLASH_MAX_NODES-1,stage_full(r.staged.len(),bytes))};
        if nodes || full {
            if full {self.recorder.borrow_mut().copies.stage_full_flushes+=1;}
            self.flush_resident_launches()?;
        }
        let address={let mut r=self.recorder.borrow_mut();let at=r.staged.len();r.staged.extend_from_slice(table);r.staged.resize(at+bytes,0);
            r.stage.as_ref().ok_or("resident stage area is absent")?.device_address()+(STAGE_BASE+at) as u64};
        let copy_bytes=if kind==Kind::Copy {table.as_chunks::<24>().0.iter().map(|row|u32::from_ne_bytes(row[16..20].try_into().unwrap()) as u64*4).sum()} else {0};
        self.append_resident_bytes(kind,build(address),grid,copy_bytes)?;Ok(true)
    }
    /// The standalone norm's three kernels with their original arguments and
    /// grids. Only ResidentNorm::apply calls this, after its extent checks.
    pub(super) fn record_resident_norm(&self,rms:crate::kernel_params::ModelRms,scale:crate::kernel_params::ModelElementwise)->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        if !self.resident_launches_active() {return Ok(false);}
        let width=rms.width as usize;
        self.append_resident(Kind::RmsPartials,encode::rms(rms,Kind::RmsPartials),[crate::launch_blocks(width.div_ceil(128)*32,128)?,rms.rows,1,128,0])?;
        let mut command=encode::rms(rms,Kind::NormTail);command.put(64,&encode::elementwise(scale).payload[..core::mem::size_of::<crate::kernel_params::ModelElementwise>()]);
        self.append_resident(Kind::NormTail,command,[crate::launch_blocks(width,128)?,rms.rows,1,128,0])?;Ok(true)
    }
    pub(super) fn record_resident_copy(&self,source:u64,target:u64,words:usize)->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        if !self.resident_launches_active() {return Ok(false);}
        if words==0 {return Ok(true);}
        let bytes=byte_count(words,4)? as u64;
        if source!=0 && source!=target && source<target+bytes && target<source+bytes {self.flush_resident_launches()?;return Ok(false);}
        if source==0 && words<=ZERO_WORDS {
            // The zero fill of a fresh output can run before every command of its
            // piece: none of them was recorded after the allocation, so none reads
            // or writes these words. One kernel then fills all such outputs.
            let fresh={let r=self.recorder.borrow();r.fresh_at==r.recorded && r.fresh.iter().any(|&(start,extent)|target>=start && target+bytes<=start+extent)};
            if fresh {
                let full={let r=self.recorder.borrow();r.zeros.len()+r.waiting.len()>=ZERO_ROWS};if full {self.flush_resident_launches()?;}
                self.context.resident_unsubmitted.set(true);self.context.pending.set(true);
                let mut r=self.recorder.borrow_mut();let at=r.calls.len();r.waiting.push((at,target,words as u32));return Ok(true);
            }
        }
        if source!=0 && self.fuse_fresh_fills(&[(source,target,words as u32)])? {return Ok(true);}
        if source==0 {self.note_zero_fill(1,1,words);}
        let kind=if source==0 {Kind::ZeroWords} else {Kind::CopyWords};let mut c=Command::new(kind);c.word(0,source);c.word(1,target);c.word(2,words as u64);
        self.append_resident(kind,c,[(words as u32).div_ceil(128),1,1,128,0])?;Ok(true)
    }
    /// Submit what was recorded, then run `direct` with recording off. Its launches and
    /// clears enter the stream in call order, after every recorded command.
    pub(super) fn unrecorded<T>(&self,direct:impl FnOnce()->Result<T,String>)->Result<T,String> {
        self.flush_resident_launches()?;let was=std::mem::replace(&mut self.recorder.borrow_mut().paused,true);
        let result=direct();self.recorder.borrow_mut().paused=was;result
    }
    /// Submit what is recorded, then pause recording (true) or resume it (false). Returns the
    /// previous paused value for the caller to restore. While paused every command launches
    /// directly, in call order, after every recorded one (as in `unrecorded`). The prompt rows pass
    /// runs this way: its pieces held about two commands each, so recording them cost more than it
    /// saved, and every new row count captured 21 to 38 new pieces.
    pub fn pause_resident_launches(&self,pause:bool)->Result<bool,String> {
        self.flush_resident_launches()?;Ok(std::mem::replace(&mut self.recorder.borrow_mut().paused,pause))
    }
    pub fn flush_resident_launches(&self)->Result<(),String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        // Every flush's tables start at one offset. Each upload is ordered after every earlier
        // reader on the one stream, as the zero rows at offset 0 are, so nothing is overwritten
        // early. An offset carried across flushes made the cut of a full area depend on history.
        if !self.recorder.borrow().paused {self.settle_fresh_fills();}
        let (calls,staged,zeros,bells,mut fed,tail)={let mut r=self.recorder.borrow_mut();if r.paused || r.calls.is_empty() && r.zeros.is_empty() {return Ok(());}r.paused=true;r.heavy=0;
            let early=std::mem::take(&mut r.early);if std::mem::take(&mut r.missed) && !early {r.ramp[2]+=1;}
            let staged=std::mem::take(&mut r.staged);r.lane_open=false;r.lane_join=false;(std::mem::take(&mut r.calls),staged,std::mem::take(&mut r.zeros),std::mem::take(&mut r.bells),r.fed,r.tail)};
        let staged_once=Cell::new(false);
        // SAFETY: Each captured node uses its original validated ABI and grid.
        // The shared device table stays aligned and owned by the recorder, which
        // drops its graphs first. Uploads and replay share one stream. Queue retention covers every indirect
        // operand through submission and the eventual real completion, including
        // the capture constructor drain. The table contains initialized fields.
        // Merged zero rows name the private zero source and fresh private outputs
        // that no other command of this piece wrote before. Their words fit the
        // source, and their row table is uploaded before this replay.
        // Staged operand tables are whole initialized bytes in the recorder's
        // private area. Their upload precedes every part of this flush on the same
        // stream, and each reader was recorded in this flush together with its table.
        // Parts keep the recorded command order on the one stream.
        let result:Result<(),String>=unsafe {(|| {
            // One part of a flush: its merged zero fill, then its commands in order.
            let submit=|calls:&[Call],zeros:&[(usize,u64,u32,bool)],bells:&[(usize,std::sync::Arc<crate::tp_mailbox_host::Bell>,u64)]|->Result<(),String> {
            if self.recorder.borrow().stage.is_none() {let stage=self.vector((COMMAND_BYTES+STAGE_BYTES)/4)?;self.recorder.borrow_mut().stage=Some(stage);}
            let mut first=None;let mut zero_rows=Vec::new();
            if let Some(grid)=zero_grid(zeros) {
                self.note_zero_fill(1,zeros.len() as u128,zeros.iter().map(|z|z.2 as usize).sum());
                // One copy kernel, first in the part, fills every fresh output from
                // the zero source. Its rows are the first bytes of the stage area.
                if self.recorder.borrow().zero_source.is_none() {let source=self.upload_vector(&vec![0.0;ZERO_WORDS])?;self.recorder.borrow_mut().zero_source=Some(source);}
                let source=self.recorder.borrow().zero_source.as_ref().ok_or("resident zero source is absent")?.device_address();
                if zeros.iter().any(|z|z.3) && self.recorder.borrow().poison_source.is_none() {
                    let poison=self.upload_vector(&vec![f32::from_bits(crate::fill_free::POISON);ZERO_WORDS])?;self.recorder.borrow_mut().poison_source=Some(poison);}
                let poison=self.recorder.borrow().poison_source.as_ref().map_or(0,|v|v.device_address());
                let rows=&mut zero_rows;
                for &(_,target,words,poisoned) in zeros {let source=if poisoned {poison} else {source};rows.extend_from_slice(&source.to_ne_bytes());rows.extend_from_slice(&target.to_ne_bytes());
                    rows.extend_from_slice(&words.to_ne_bytes());rows.extend_from_slice(&0u32.to_ne_bytes());}
                let address=self.recorder.borrow().stage.as_ref().ok_or("resident stage area is absent")?.device_address()+COMMAND_BYTES as u64;
                first=Some(Call {grid,copy_bytes:zeros.iter().map(|z|z.2 as u64*4).sum(),command:encode::copy(crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:address})});
            }
            let count=calls.len()+usize::from(first.is_some());
            let table_bytes=count*core::mem::size_of::<Command>();
            let table_offset=COMMAND_BYTES.checked_sub(table_bytes).ok_or("resident command table exceeds capacity")?;
            let table_address=self.recorder.borrow().stage.as_ref().ok_or("resident stage area is absent")?.device_address()+table_offset as u64;
            let key=first.iter().chain(calls).map(|c|c.grid).collect::<Vec<_>>();
            // SAFETY: The table is private aligned persistent storage. Each
            // replay node names one exact command index and the original
            // validated grid. Capture allocates no storage or reads results.
            // All indirect operands remain retained while unsubmitted is
            // true, including the capture constructor's real stream drain.
            // A side-branch command forks from the capturing stream at its place and is launched on the second
            // stream; every branch joins after the last command, so the piece is one graph that ends when all of it
            // has run. The events live until the capture has ended.
            if first.iter().chain(calls).any(|call|call.grid.kind==Kind::ShadowSide) && self.recorder.borrow().side.is_none() {
                let side=Stream::new(self.context)?;self.recorder.borrow_mut().side=Some(side);}
            let side=self.recorder.borrow().side.as_ref().map(|stream|stream.raw);
            // PROJ-DELIVERY side lane: a run of SIDE calls waits on an event recorded on the capture stream at its place and
            // launches on the lane stream; a JOIN call, and the end of the piece, wait on an event recorded on the lane stream
            // after its last call. The graph keeps those edges.
            let fork=self.recorder.borrow().lane_fork.as_ref().map(|(stream,fork,join)|(stream.raw,fork.raw,join.raw));
            // The launches of this piece, into an open capture. `events` holds the side-branch events of that capture; they
            // live until it has ended, and the list is emptied then.
            let events=RefCell::new(Vec::new());
            let launches=||->Result<(),String> {
                let (mut forked,mut pending)=(false,false);
                let join=|pending:&mut bool|->Result<(),String> {if !*pending {return Ok(());}*pending=false;let (lane,_,done)=fork.ok_or("resident side lane stream is absent")?;
                    check(cuEventRecord(done,lane),"resident side lane join record")?;check(cuStreamWaitEvent(self.stream.raw,done,0),"resident side lane join wait")};
                for (index,call) in first.iter().chain(calls).enumerate() {let g=call.grid;let (mut address,mut index)=(table_address,index as u64);
                    if g.kind==Kind::ShadowSide {
                        let side=side.ok_or("resident side stream is absent")?;
                        let fork=Event::record(&self.stream)?;check(cuStreamWaitEvent(side,fork.raw,0),"resident side fork")?;
                        let function=self.module.function(g.kind.kernel())?;self.context.receipt_launch(function,g.threads,g.shared)?;
                        self.context.check_execution(launch_kernel(false,function,[g.blocks,g.rows,g.planes],g.threads,g.shared,side,[parameter(&mut address),parameter(&mut index)].as_mut_ptr()),"resident side launch")?;
                        let done=Event::record_raw(self.context,side)?;events.borrow_mut().push(fork);events.borrow_mut().push(done);continue;
                    }
                    if g.lane==SIDE {let (lane,start,_)=fork.ok_or("resident side lane stream is absent")?;
                        if !forked {check(cuEventRecord(start,self.stream.raw),"resident side lane fork record")?;check(cuStreamWaitEvent(lane,start,0),"resident side lane fork wait")?;forked=true;pending=true;}
                        self.side_launch.set(lane);
                    } else {forked=false;if g.lane==JOIN {join(&mut pending)?;}}
                    let launched=self.launch_flash_grid(g.kind.kernel(),g.blocks,g.rows,g.planes,g.threads,g.shared,&mut [parameter(&mut address),parameter(&mut index)]);
                    self.side_launch.set(ptr::null_mut());launched?;
                }
                join(&mut pending)?;
                for done in events.borrow().iter().skip(1).step_by(2) {check(cuStreamWaitEvent(self.stream.raw,done.raw,0),"resident side join")?;}
                Ok(())
            };
            // A new executable for this key: room in the retained set first, then a capture (it begins with a stream wait).
            let capture=||->Result<(),String> {
                let storage=crate::GraphAllowance::flash(count)?.maximum_bytes();
                let mut retired=false;
                loop {
                    let r=self.recorder.borrow();
                    let retained=r.pieces.values().map(|piece|piece.graph.storage_bytes).sum::<usize>();
                    let fits=retained.saturating_add(storage)<=128*DecodeGraph::MAX_RESERVATION_BYTES;
                    let evict=!fits && !r.pieces.is_empty();drop(r);if !evict {break;}
                    // Retire the executable used longest ago, after its replays have ended.
                    if !retired {self.synchronize()?;retired=true;}let mut r=self.recorder.borrow_mut();
                    if let Some(old)=r.pieces.iter().min_by_key(|(_,piece)|piece.used).map(|(key,_)|key.clone()) {r.pieces.remove(&old);r.copies.piece_evictions+=1;}
                }
                let graph=self.capture_batch_bounded(count,launches);events.borrow_mut().clear();let graph=graph?;
                let mut r=self.recorder.borrow_mut();r.pieces.insert(key.clone(),Piece {graph,used:0});
                // Print only: was this key captured before, and evicted since?
                let hash=crate::resident_launch::capture_keys::hash(key.iter().map(|g|[g.kind as u32,g.blocks,g.rows,g.planes,g.threads,g.shared]));
                if r.seen_keys.note(hash) {r.copies.captures_new_key+=1;} else {r.copies.captures_seen_key+=1;}Ok(())
            };
            // The retained executable of this key runs the piece. When the key is absent and `SAME_KERNELS` executables of
            // the same kernels are retained, the one used longest ago is the donor, if its replays have ended (used at or
            // before the last real stream wait): it is updated in place below, after this piece's table upload. A launch
            // whose extent follows the context (a row count, a zero fill) then costs an update of one executable, not the
            // capture of a new one with its stream wait, and the retained set does not grow by a piece a step. With no
            // donor, a new executable is captured now.
            let donor=if self.recorder.borrow().pieces.contains_key(&key) {None} else {
                let r=self.recorder.borrow();let (mut same,mut oldest)=(0usize,None::<(&Vec<Grid>,u64)>);
                for (old,piece) in r.pieces.iter().filter(|(old,_)|same_kernels(old,&key)) {same+=1;if oldest.is_none_or(|(_,used)|piece.used<used) {oldest=Some((old,piece.used));}}
                oldest.filter(|&(_,used)|same>=SAME_KERNELS && used<=r.drained_at).map(|(old,_)|old.clone())
            };
            if donor.is_none() && !self.recorder.borrow().pieces.contains_key(&key) {capture()?;}
            let replay=||->Result<(),String> {
                let mut piece=self.recorder.borrow_mut().pieces.remove(&key).ok_or("resident piece disappeared")?;
                {let mut r=self.recorder.borrow_mut();r.clock+=1;piece.used=r.clock;}
                // SAFETY: The executable owns only its persistent table address.
                // All table-referenced operands are held by queue deferred frees
                // until this replay has actually completed. Table replacement is
                // ordered after prior replay. Captured grids and ABI are unchanged.
                let result=self.replay_decode(&piece.graph);
                self.recorder.borrow_mut().pieces.insert(key.clone(),piece);result
            };
            let mut bytes=Vec::with_capacity(count*core::mem::size_of::<Command>());
            for call in first.iter().chain(calls) {bytes.extend_from_slice(&call.command.kind.to_ne_bytes());for word in call.command.reserved {bytes.extend_from_slice(&word.to_ne_bytes());}bytes.extend_from_slice(&call.command.payload);}
            // One contiguous upload carries commands, zero rows and operand tables.
            // All earlier readers finish on this stream before any byte is replaced.
            let payload=if staged_once.replace(true) {&[]} else {staged.as_slice()};
            if !zero_rows.is_empty() || !payload.is_empty() {
                bytes.resize(table_bytes+ZERO_BYTES,0);bytes[table_bytes..table_bytes+zero_rows.len()].copy_from_slice(&zero_rows);bytes.extend_from_slice(payload);
            }
            // write_buffer's immutable pinned staging stays retained until the
            // actual drain, and upload/replay/next upload share one stream.
            let stage=self.recorder.borrow_mut().stage.take().ok_or("resident stage area is absent")?;
            let uploaded=self.write_buffer(&stage.buffer,table_offset,&bytes);self.recorder.borrow_mut().stage=Some(stage);
            self.context.pending.set(true);
            let result=uploaded.and_then(|_| {
                let Some(old)=donor else {return replay();};
                // The donor takes this piece's grids in place: a capture with no stream wait (it launches nothing), then an
                // update of the executable, which changes only its later launches, then its replay (`end_segment` does the
                // three). The capture needs free upload staging and funding for its source graph; without them the donor
                // stays as it is and a new executable is captured the usual way. The table is uploaded already.
                let mut piece=self.recorder.borrow_mut().pieces.remove(&old).ok_or("resident donor piece disappeared")?;
                if !self.begin_segment(count.saturating_sub(32).div_ceil(64).max(1),1)? {
                    self.recorder.borrow_mut().pieces.insert(old,piece);capture()?;return replay();
                }
                if let Err(error)=launches() {self.abort_segment();events.borrow_mut().clear();self.recorder.borrow_mut().pieces.insert(old,piece);return Err(error);}
                let mut slot=Some(piece.graph);let ended=self.end_segment(&mut slot);events.borrow_mut().clear();
                let mut r=self.recorder.borrow_mut();
                match (ended,slot) {
                    (Ok(updated),Some(graph))=>{
                        r.clock+=1;piece=Piece {graph,used:r.clock};r.pieces.insert(key.clone(),piece);
                        // Not updated: the driver refused the update and `end_segment` put a new executable in the donor's place.
                        if updated {r.updates+=1;} else {
                            let hash=crate::resident_launch::capture_keys::hash(key.iter().map(|g|[g.kind as u32,g.blocks,g.rows,g.planes,g.threads,g.shared]));
                            if r.seen_keys.note(hash) {r.copies.captures_new_key+=1;} else {r.copies.captures_seen_key+=1;}
                        }Ok(())
                    },
                    // The update or the replay failed. An executable that is left stays retained under the donor's key.
                    (Err(error),Some(graph))=>{r.pieces.insert(old,Piece {graph,used:0});Err(error)},
                    (Err(error),None)=>Err(error),
                    (Ok(_),None)=>Err("resident piece update left no executable".into()),
                }
            });
            if result.is_ok() {let mut r=self.recorder.borrow_mut();
                for call in first.iter().chain(calls) {let index=match call.grid.kind {Kind::Copy=>0,Kind::CopyWords=>1,Kind::ZeroWords=>2,_=>continue};
                    r.copies.launches[index]+=1;r.copies.bytes[index]+=call.copy_bytes;}
                if let Some(call)=first {r.copies.merged_zero_launches+=1;r.copies.merged_zero_bytes+=call.copy_bytes;}
                for (_,bell,sequence) in bells {bell.queued(*sequence);}
            }result
            };
            let (mut start,mut zero_at,mut bell_at)=(0usize,0usize,0usize);
            while start<calls.len() || zero_at<zeros.len() {
                let alone=leaves_alone(fed,tail,first_arithmetic(&calls[start..]));
                let limit=if alone {1} else {part_limit(fed,tail)};let (mut end,mut count)=(start,0usize);
                // A part never ends between a side call and its join.
                let mut open=false;
                while end<calls.len() {let g=calls[end].grid;match g.lane {SIDE=>open=true,JOIN=>open=false,_=>{}}end+=1;if arithmetic(g.kind) {count+=1;if count>=limit && !open {break;}}}
                // A zero fill belongs to the part of the next recorded command.
                let mut zero_end=zero_at;while zero_end<zeros.len() && (zeros[zero_end].0<end || end==calls.len()) {zero_end+=1;}
                let mut bell_end=bell_at;while bell_end<bells.len() && bells[bell_end].0<end {bell_end+=1;}
                submit(&calls[start..end],&zeros[zero_at..zero_end],&bells[bell_at..bell_end])?;
                start=end;zero_at=zero_end;bell_at=bell_end;fed=if tail {fed.saturating_add(count)} else if alone {fed} else {fed+1};
            }Ok(())
        })()};
        self.recorder.borrow_mut().fed=fed;
        self.context.resident_unsubmitted.set(false);self.recorder.borrow_mut().paused=false;
        if let Err(error)=result {
            // Accepted producers cannot be retried after a partial submission.
            // Drain earlier work before retiring the queue and its dependent values.
            let _drained=self.synchronize();self.recorder.borrow_mut().failure=Some(error.clone());Err(error)
        } else {Ok(())}
    }
}
#[cfg(test)] mod ramp_tests {
    use super::{part_limit,RAMP};
    /// An executable is updated in place only for a piece of the same kernels, block sizes and lanes; grids may differ.
    #[test] fn a_donor_piece_has_the_same_kernels_block_sizes_and_lanes() {
        use super::{same_kernels,Grid,Kind,MAIN,SIDE,JOIN};
        let call=|kind,blocks,lane|Grid {kind,blocks,rows:1,planes:1,threads:128,shared:0,lane};
        let piece=[call(Kind::Copy,64,MAIN),call(Kind::Plain,36,SIDE),call(Kind::CausalFast,32,JOIN),call(Kind::CopyWords,4,MAIN)];
        // Other grids only (blocks, rows, planes): the same graph with other launch extents.
        let mut grids=piece;grids[0].blocks=128;grids[0].rows=72;grids[2].blocks=96;grids[3].planes=2;assert!(same_kernels(&piece,&grids));
        // A call that moved to another lane changes the fork and join edges of the graph: not a donor.
        for lane in [MAIN,JOIN] {let mut moved=piece;moved[1].lane=lane;assert!(!same_kernels(&piece,&moved));}
        let mut joined=piece;joined[2].lane=MAIN;assert!(!same_kernels(&piece,&joined));
        // Another kernel, block size, shared size or length: not a donor.
        let mut other=piece;other[1].kind=Kind::PlainBf16;assert!(!same_kernels(&piece,&other));
        let mut other=piece;other[1].threads=160;assert!(!same_kernels(&piece,&other));
        let mut other=piece;other[1].shared=30720;assert!(!same_kernels(&piece,&other));
        assert!(!same_kernels(&piece,&piece[..3]) && same_kernels(&piece,&piece));
    }
    /// The parts of one flush after a wait, as `part_limit` alone cuts them (the weight batch that `leaves_alone` sends
    /// out by itself is not modelled here). `tail`: the wait was a selection read.
    fn parts(mut fed:usize,mut arithmetic:usize,tail:bool)->(Vec<usize>,usize) {
        let mut out=Vec::new();while arithmetic>0 {let part=part_limit(fed,tail).min(arithmetic);out.push(part);arithmetic-=part;fed=if tail {fed+part} else {fed+1};}(out,fed)
    }
    #[test] fn only_a_weight_batch_leaves_alone_as_the_second_part() {
        use super::{leaves_alone,Kind};
        for kind in [Kind::MixedBf16,Kind::Mixed,Kind::PlainBatch] {assert!(leaves_alone(1,false,Some(kind)));assert!(!leaves_alone(1,true,Some(kind)));
            for fed in [0,2,3,4,usize::MAX] {assert!(!leaves_alone(fed,false,Some(kind)));}}
        // A chain's first member, a norm or an expert launch there keeps the limit of four.
        for kind in [Kind::CausalFast,Kind::CausalChain,Kind::Causal,Kind::Expert,Kind::Mx,Kind::PlainBf16,Kind::Plain] {assert!(!leaves_alone(1,false,Some(kind)));}
        assert!(!leaves_alone(1,false,None));
    }
    #[test] fn the_ramp_counts_commands_only_after_a_selection_read() {
        // After a wait that is not a selection read the ramp is the first design: it counts parts (1, 4, 16, 64, then no limit).
        assert_eq!(parts(0,400,false).0,[1,4,16,64,315]);
        for (fed,limit) in [(0,1),(1,4),(2,16),(3,64),(4,usize::MAX),(usize::MAX,usize::MAX)] {assert_eq!(part_limit(fed,false),limit);}
        // There a part that is forced out early ends a step of the ramp, as on main: 1 forced, 2 forced, then 16.
        assert_eq!(part_limit(2,false),16);
        // After a selection read, with nothing forced out: each part is as large as all before it, until the host is far ahead.
        assert_eq!(parts(0,400,true).0,[1,1,2,4,8,16,32,64,272]);
        // An upload forces out parts of 1, 1, 2 and 2 commands. The ramp goes on from 6 commands: the layer's next commands
        // leave in parts of 4, 8, 16, 32 and 64, not as one part of all that follows. (Counting parts, the ramp is over here.)
        let fed:usize=[1,1,2,2].iter().sum();assert_eq!(parts(fed,70,true).0,[4,8,16,32,10]);assert_eq!(parts(4,70,false).0,[70]);
        // A part with no arithmetic command (copies only) does not move the ramp.
        assert_eq!(parts(0,0,true),(vec![],0));assert_eq!(part_limit(0,true),1);
        for fed in 1..RAMP {let limit=part_limit(fed,true);assert!(limit.is_power_of_two() && limit<=fed && 2*limit>fed,"fed {fed}");}
        assert_eq!(part_limit(RAMP,true),usize::MAX);assert_eq!(part_limit(usize::MAX,true),usize::MAX);
    }
}
