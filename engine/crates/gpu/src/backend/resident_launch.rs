//! Queue-retained operands and a stable indirect table permit larger graph pieces.
use super::*;
use crate::{resident_launch::{Command,Kind},resident_launch_encode as encode};
use std::collections::BTreeMap;
#[derive(Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
struct Grid {kind:Kind,blocks:u32,rows:u32,planes:u32,threads:u32,shared:u32}
struct Call {grid:Grid,command:Command,copy_bytes:u64}
// The executable drops before the table. Queued table drops retain its storage.
struct Piece<'a> {graph:DecodeGraph<'a>,table:DeviceVector<'a>,used:u64}
/// After a real wait the GPU is idle. The recorder then submits a short first
/// part and longer later parts, so the GPU works while the host records. The
/// limits count arithmetic commands and depend only on the recorded order.
fn part_limit(fed:usize)->usize {match fed {0=>1,1=>4,2=>16,3=>64,_=>usize::MAX}}
fn arithmetic(kind:Kind)->bool {matches!(kind,Kind::Causal|Kind::Expert|Kind::Plain|Kind::PlainBf16|Kind::Mx|Kind::Mixed|Kind::NvStream|Kind::CausalFast|Kind::NvTile|Kind::DeviceRoute)}
fn zero_grid(zeros:&[(usize,u64,u32)])->Option<Grid> {
    let longest=zeros.iter().map(|z|z.2).max()?;Some(Grid {kind:Kind::Copy,blocks:longest.div_ceil(128),rows:zeros.len() as u32,planes:1,threads:128,shared:0})
}
/// Operand tables of recorded commands share one private device area. One upload
/// per piece carries them, so a join or a batch does not end the piece.
use crate::resident_launch::stage::{BYTES as STAGE_BYTES,BASE as STAGE_BASE,full as stage_full};
/// Zero fills of fresh private outputs share one copy kernel at the start of a
/// piece. Its rows read one zero source. The row table starts the stage area.
const ZERO_ROWS:usize=crate::resident_launch::stage::ZERO_ROWS;const ZERO_WORDS:usize=1<<15;
#[derive(Default)]
pub(super) struct Recorder<'a> {enabled:bool,paused:bool,pub(super) failure:Option<String>,calls:Vec<Call>,pieces:BTreeMap<Vec<Grid>,Piece<'a>>,
    stage:Option<DeviceVector<'a>>,staged:Vec<u8>,
    zero_source:Option<DeviceVector<'a>>,zeros:Vec<(usize,u64,u32)>,fresh:Vec<(u64,u64)>,fresh_at:u64,recorded:u64,
    heavy:usize,fed:usize,clock:u64,copies:crate::resident_launch::CopyCounts,seen_keys:crate::resident_launch::capture_keys::Seen}
impl Recorder<'_> {
    /// A real stream wait has ended: nothing is queued on the device.
    pub(super) fn drained(&mut self) {self.fed=0;}
}
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
    pub fn feed_resident_launches(&self)->Result<(),String> {self.flush_resident_launches()?;self.recorder.borrow_mut().drained();Ok(())}
    pub fn resident_launches_active(&self)->bool {let r=self.recorder.borrow();r.enabled && !r.paused}
    pub(super) fn disable_resident_launches(&self)->Result<(),String> {let result=self.flush_resident_launches();self.recorder.borrow_mut().enabled=false;result}
    pub fn resident_copy_counts(&self)->crate::resident_launch::CopyCounts {self.recorder.borrow().copies}
    fn append_resident(&self,kind:Kind,command:Command,grid:[u32;5])->Result<(),String> {
        let words=match kind {Kind::CopyWords|Kind::ZeroWords=>u64::from_ne_bytes(command.payload[16..24].try_into().unwrap()),
            Kind::Copy=>u32::from_ne_bytes(command.payload[16..20].try_into().unwrap()) as u64,_=>0};
        self.append_resident_bytes(kind,command,grid,words.saturating_mul(4))
    }
    fn append_resident_bytes(&self,kind:Kind,command:Command,grid:[u32;5],copy_bytes:u64)->Result<(),String> {
        if grid[0]==0 {return Ok(());}
        if grid[1]==0 || grid[1]>65535 || grid[2]==0 || grid[2]>65535 || grid[3]==0 || grid[3]>1024 || grid[3]%32!=0 {return Err("resident recorded grid exceeds driver limits".into());}
        if self.recorder.borrow().calls.len()>=crate::GraphAllowance::FLASH_MAX_NODES-1 {self.flush_resident_launches()?;}
        self.context.resident_unsubmitted.set(true);self.context.pending.set(true);self.stats.borrow_mut().recorded_kinds[kind as usize]+=1;
        let mut r=self.recorder.borrow_mut();r.recorded=r.recorded.wrapping_add(1);
        r.calls.push(Call {grid:Grid {kind,blocks:grid[0],rows:grid[1],planes:grid[2],threads:grid[3],shared:grid[4]},command,copy_bytes});
        if arithmetic(kind) {r.heavy+=1;
            // Submit this part now only when its executable exists: a capture
            // must first drain the stream, and that would stop the device.
            if r.heavy==part_limit(r.fed) {let key=zero_grid(&r.zeros).into_iter().chain(r.calls.iter().map(|c|c.grid)).collect::<Vec<_>>();
                if r.pieces.contains_key(&key) {drop(r);return self.flush_resident_launches();}}
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
    /// Called only with the original checked kernel ABI at the GPU boundary.
    pub(super) unsafe fn record_resident_launch(&self,name:&str,grid:[u32;5],args:&[*mut c_void])->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        if !self.resident_launches_active() {return Ok(false);}
        // SAFETY: launch_flash_grid's caller supplies this registered name's
        // original ABI pointers. These typed reads copy live argument values;
        // serializers copy numeric fields separately, never Rust padding. No
        // borrowed host pointer enters the retained command. Queue-owned GPU
        // operands survive all deferred submission and completion boundaries.
        let command=unsafe {match name {
            "causal_operation"=>Some((Kind::Causal,encode::causal(args[0].cast::<crate::causal_ops::Arguments>().read_unaligned()))),
            "expert_operation"=>Some((Kind::Expert,encode::expert(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned()))),
            "causal_fast"=>Some((Kind::CausalFast,encode::causal_fast(args[0].cast::<crate::causal_ops::Arguments>().read_unaligned()))),
            "nv_rows_stream"=>Some((Kind::NvStream,encode::nv_stream(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned()))),
            "nv_tile"=>Some((Kind::NvTile,encode::nv_tile(args[0].cast::<crate::expert_ops::LaunchArguments>().read_unaligned()))),
            "device_route_build"=>Some((Kind::DeviceRoute,encode::device_route(args[0].cast::<crate::device_route::Build>().read_unaligned()))),
            "flash_next_mxfp8_m1"=>Some((Kind::Mx,encode::mx(args[0].cast::<crate::next_params::MxLayer>().read_unaligned()))),
            "flash_resident_projection_batch"=>Some((Kind::Mixed,encode::mixed(args[0].cast::<crate::next_params::ResidentProjectionBatch>().read_unaligned()))),
            "bf16_lanes_round"=>Some((Kind::Round,encode::round(args[0].cast::<crate::mhc_boundary::LaneCast>().read_unaligned()))),
            "flash_hybrid_copy"=>Some((Kind::Copy,encode::copy(args[0].cast::<crate::kernel_params::HybridCopy>().read_unaligned()))),
            "mhc_prepare"=>Some((Kind::Prepare,encode::prepare(args[0].cast::<crate::mhc_params::Prepare>().read_unaligned()))),
            "mhc_bf16_collapse"=>Some((Kind::Collapse,encode::collapse(args[0].cast::<crate::mhc_params::Collapse>().read_unaligned()))),
            "mhc_bf16_combine"=>Some((Kind::Combine,encode::combine(args[0].cast::<crate::mhc_params::Combine>().read_unaligned()))),
            "model_rms_partials"=>Some((Kind::RmsPartials,encode::rms(args[0].cast::<crate::kernel_params::ModelRms>().read_unaligned(),Kind::RmsPartials))),
            "model_rms_apply"=>Some((Kind::RmsApply,encode::rms(args[0].cast::<crate::kernel_params::ModelRms>().read_unaligned(),Kind::RmsApply))),
            "model_elementwise"=>Some((Kind::Elementwise,encode::elementwise(args[0].cast::<crate::kernel_params::ModelElementwise>().read_unaligned()))),
            "project_f32_ordered"=>{if args.len()!=5 {return Err("ordered projection argument count differs".into());}let mut c=Command::new(Kind::Plain);for i in 0..3 {c.word(i,args[i].cast::<u64>().read_unaligned());}for i in 3..5 {c.word(i,args[i].cast::<u32>().read_unaligned() as u64);}Some((Kind::Plain,c))},
            "project_bf16_ordered"=>{if args.len()!=5 {return Err("ordered BF16 projection argument count differs".into());}let mut c=Command::new(Kind::PlainBf16);for i in 0..3 {c.word(i,args[i].cast::<u64>().read_unaligned());}for i in 3..5 {c.word(i,args[i].cast::<u32>().read_unaligned() as u64);}Some((Kind::PlainBf16,c))},
            "tp_mailbox_put_wide"=>{if args.len()!=8 {return Err("resident mailbox argument count differs".into());}let mut c=Command::new(Kind::PutWide);for (i,p) in args.iter().enumerate() {c.word(i,p.cast::<u64>().read_unaligned());}Some((Kind::PutWide,c))},
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
        if self.recorder.borrow().stage.is_none() {let stage=self.vector(STAGE_BYTES/4)?;self.recorder.borrow_mut().stage=Some(stage);}
        let (nodes,full)={let r=self.recorder.borrow();(r.calls.len()>=crate::GraphAllowance::FLASH_MAX_NODES-1,stage_full(r.staged.len(),bytes))};
        if nodes || full {
            if full {self.recorder.borrow_mut().copies.stage_full_flushes+=1;}
            self.flush_resident_launches()?;
        }
        let address={let mut r=self.recorder.borrow_mut();let at=r.staged.len();r.staged.extend_from_slice(table);r.staged.resize(at+bytes,0);
            r.stage.as_ref().ok_or("resident stage area is absent")?.device_address()+(STAGE_BASE+at) as u64};
        let copy_bytes=if kind==Kind::Copy {table.chunks_exact(24).map(|row|u32::from_ne_bytes(row[16..20].try_into().unwrap()) as u64*4).sum()} else {0};
        self.append_resident_bytes(kind,build(address),grid,copy_bytes)?;Ok(true)
    }
    /// The standalone norm's three kernels with their original arguments and
    /// grids. Only ResidentNorm::apply calls this, after its extent checks.
    pub(super) fn record_resident_norm(&self,rms:crate::kernel_params::ModelRms,scale:crate::kernel_params::ModelElementwise)->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        if !self.resident_launches_active() {return Ok(false);}
        let width=rms.width as usize;
        self.append_resident(Kind::RmsPartials,encode::rms(rms,Kind::RmsPartials),[crate::launch_blocks(width.div_ceil(128)*32,128)?,rms.rows,1,128,0])?;
        self.append_resident(Kind::RmsApply,encode::rms(rms,Kind::RmsApply),[crate::launch_blocks(width,128)?,rms.rows,1,128,0])?;
        self.append_resident(Kind::Elementwise,encode::elementwise(scale),[crate::launch_blocks(scale.elements as usize,128)?,1,1,128,0])?;Ok(true)
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
                if self.recorder.borrow().zeros.len()>=ZERO_ROWS {self.flush_resident_launches()?;}
                self.context.resident_unsubmitted.set(true);self.context.pending.set(true);
                let mut r=self.recorder.borrow_mut();let at=r.calls.len();r.zeros.push((at,target,words as u32));return Ok(true);
            }
        }
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
    pub(super) fn flush_resident_launches(&self)->Result<(),String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        // Every flush's tables start at one offset. Each upload is ordered after every earlier
        // reader on the one stream, as the zero rows at offset 0 are, so nothing is overwritten
        // early. An offset carried across flushes made the cut of a full area depend on history.
        let (calls,staged,zeros,mut fed)={let mut r=self.recorder.borrow_mut();if r.paused || r.calls.is_empty() && r.zeros.is_empty() {return Ok(());}r.paused=true;r.heavy=0;
            let staged=std::mem::take(&mut r.staged);(std::mem::take(&mut r.calls),staged,std::mem::take(&mut r.zeros),r.fed)};
        // SAFETY: Each captured node uses its original validated ABI and grid.
        // The private table stays aligned and owned with its executable; uploads
        // and replay share one stream. Queue retention covers every indirect
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
            let submit=|calls:&[Call],zeros:&[(usize,u64,u32)]|->Result<(),String> {
            let mut first=None;
            if let Some(grid)=zero_grid(zeros) {
                self.note_zero_fill(1,zeros.len() as u128,zeros.iter().map(|z|z.2 as usize).sum());
                // One copy kernel, first in the part, fills every fresh output from
                // the zero source. Its rows are the first bytes of the stage area.
                if self.recorder.borrow().zero_source.is_none() {let source=self.upload_vector(&vec![0.0;ZERO_WORDS])?;self.recorder.borrow_mut().zero_source=Some(source);}
                if self.recorder.borrow().stage.is_none() {let stage=self.vector(STAGE_BYTES/4)?;self.recorder.borrow_mut().stage=Some(stage);}
                let source=self.recorder.borrow().zero_source.as_ref().ok_or("resident zero source is absent")?.device_address();
                let mut rows=Vec::with_capacity(zeros.len()*24);
                for &(_,target,words) in zeros {rows.extend_from_slice(&source.to_ne_bytes());rows.extend_from_slice(&target.to_ne_bytes());
                    rows.extend_from_slice(&words.to_ne_bytes());rows.extend_from_slice(&0u32.to_ne_bytes());}
                let stage=self.recorder.borrow_mut().stage.take().ok_or("resident stage area is absent")?;
                let uploaded=self.write_buffer(&stage.buffer,0,&rows);let address=stage.device_address();self.recorder.borrow_mut().stage=Some(stage);uploaded?;
                first=Some(Call {grid,copy_bytes:zeros.iter().map(|z|z.2 as u64*4).sum(),command:encode::copy(crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:address})});
            }
            let count=calls.len()+usize::from(first.is_some());
            let key=first.iter().chain(calls).map(|c|c.grid).collect::<Vec<_>>();
            if !self.recorder.borrow().pieces.contains_key(&key) {
                if self.recorder.borrow().pieces.len()>=128 {
                    // Retire the executable used longest ago, after its replays have ended.
                    self.synchronize()?;let mut r=self.recorder.borrow_mut();
                    if let Some(old)=r.pieces.iter().min_by_key(|(_,piece)|piece.used).map(|(key,_)|key.clone()) {r.pieces.remove(&old);r.copies.piece_evictions+=1;}
                }
                let table=self.vector(count*core::mem::size_of::<Command>()/4)?;
                // SAFETY: The table is private aligned persistent storage. Each
                // replay node names one exact command index and the original
                // validated grid. Capture allocates no storage or reads results.
                // All indirect operands remain retained while unsubmitted is
                // true, including the capture constructor's real stream drain.
                let graph=self.capture_batch_bounded(count,|| {
                    for (index,call) in first.iter().chain(calls).enumerate() {let g=call.grid;let (mut address,mut index)=(table.device_address(),index as u64);
                        self.launch_flash_grid(g.kind.kernel(),g.blocks,g.rows,g.planes,g.threads,g.shared,&mut [parameter(&mut address),parameter(&mut index)])?;
                    }Ok(())
                })?;
                let mut r=self.recorder.borrow_mut();r.pieces.insert(key.clone(),Piece {graph,table,used:0});
                // Print only: was this key captured before, and evicted since?
                let hash=crate::resident_launch::capture_keys::hash(key.iter().map(|g|[g.kind as u32,g.blocks,g.rows,g.planes,g.threads,g.shared]));
                if r.seen_keys.note(hash) {r.copies.captures_new_key+=1;} else {r.copies.captures_seen_key+=1;}drop(r);
            }
            let mut piece=self.recorder.borrow_mut().pieces.remove(&key).ok_or("resident piece disappeared")?;
            {let mut r=self.recorder.borrow_mut();r.clock+=1;piece.used=r.clock;}
            let mut bytes=Vec::with_capacity(count*core::mem::size_of::<Command>());
            for call in first.iter().chain(calls) {bytes.extend_from_slice(&call.command.kind.to_ne_bytes());for word in call.command.reserved {bytes.extend_from_slice(&word.to_ne_bytes());}bytes.extend_from_slice(&call.command.payload);}
            // write_buffer's immutable pinned staging stays retained until the
            // actual drain, and upload/replay/next upload share one stream.
            let uploaded=self.write_buffer(&piece.table.buffer,0,&bytes);
            self.context.pending.set(true);
            // SAFETY: The executable owns only its persistent table address.
            // All table-referenced operands are held by queue deferred frees
            // until this replay has actually completed. Table replacement is
            // ordered after prior replay. Captured grids and ABI are unchanged.
            let result=uploaded.and_then(|_|self.replay_decode(&piece.graph));
            self.recorder.borrow_mut().pieces.insert(key,piece);
            if result.is_ok() {let mut r=self.recorder.borrow_mut();
                for call in first.iter().chain(calls) {let index=match call.grid.kind {Kind::Copy=>0,Kind::CopyWords=>1,Kind::ZeroWords=>2,_=>continue};
                    r.copies.launches[index]+=1;r.copies.bytes[index]+=call.copy_bytes;}
                if let Some(call)=first {r.copies.merged_zero_launches+=1;r.copies.merged_zero_bytes+=call.copy_bytes;}
            }result
            };
            if !staged.is_empty() {
                let stage=self.recorder.borrow_mut().stage.take().ok_or("resident stage area is absent")?;
                let uploaded=self.write_buffer(&stage.buffer,STAGE_BASE,&staged);self.recorder.borrow_mut().stage=Some(stage);uploaded?;
            }
            let (mut start,mut zero_at)=(0usize,0usize);
            while start<calls.len() || zero_at<zeros.len() {
                let limit=part_limit(fed);let (mut end,mut count)=(start,0usize);
                while end<calls.len() {let heavy=arithmetic(calls[end].grid.kind);end+=1;if heavy {count+=1;if count>=limit {break;}}}
                // A zero fill belongs to the part of the next recorded command.
                let mut zero_end=zero_at;while zero_end<zeros.len() && (zeros[zero_end].0<end || end==calls.len()) {zero_end+=1;}
                submit(&calls[start..end],&zeros[zero_at..zero_end])?;
                start=end;zero_at=zero_end;fed+=1;
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
