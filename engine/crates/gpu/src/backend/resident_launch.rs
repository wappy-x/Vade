//! Queue-retained operands and a stable indirect table permit larger graph pieces.
use super::*;
use crate::{resident_launch::{Command,Kind},resident_launch_encode as encode};
use std::collections::BTreeMap;
#[derive(Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
struct Grid {kind:Kind,blocks:u32,rows:u32,planes:u32,threads:u32,shared:u32}
struct Call {grid:Grid,command:Command}
// The executable drops before the table. Queued table drops retain its storage.
struct Piece<'a> {graph:DecodeGraph<'a>,table:DeviceVector<'a>}
#[derive(Default)]
pub(super) struct Recorder<'a> {enabled:bool,paused:bool,pub(super) failure:Option<String>,calls:Vec<Call>,pieces:BTreeMap<Vec<Grid>,Piece<'a>>}
impl<'a> Backend<'a> {
    /// Only the owned GLM queue enables recording. Ordinary callers stay eager.
    pub fn enable_resident_launches(&self)->Result<(),String> {
        self.require_resident_stream()?;
        if !self.resident_queue_active() {return Err("resident launch recording requires an owned queue".into());}
        self.require_arm_kernels(&["resident_launch_causal","resident_launch_copy_words","resident_launch_zero_words"])?;
        self.recorder.borrow_mut().enabled=true;Ok(())
    }
    pub fn resident_launches_active(&self)->bool {let r=self.recorder.borrow();r.enabled && !r.paused}
    pub(super) fn disable_resident_launches(&self)->Result<(),String> {let result=self.flush_resident_launches();self.recorder.borrow_mut().enabled=false;result}
    fn append_resident(&self,kind:Kind,command:Command,grid:[u32;5])->Result<(),String> {
        if grid[0]==0 {return Ok(());}
        if grid[1]==0 || grid[1]>65535 || grid[2]==0 || grid[2]>65535 || grid[3]==0 || grid[3]>1024 || grid[3]%32!=0 {return Err("resident recorded grid exceeds driver limits".into());}
        if self.recorder.borrow().calls.len()>=crate::GraphAllowance::FLASH_MAX_NODES {self.flush_resident_launches()?;}
        self.context.resident_unsubmitted.set(true);self.context.pending.set(true);
        self.recorder.borrow_mut().calls.push(Call {grid:Grid {kind,blocks:grid[0],rows:grid[1],planes:grid[2],threads:grid[3],shared:grid[4]},command});Ok(())
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
            "flash_next_mxfp8_m1"=>Some((Kind::Mx,encode::mx(args[0].cast::<crate::next_params::MxLayer>().read_unaligned()))),
            "bf16_lanes_round"=>Some((Kind::Round,encode::round(args[0].cast::<crate::mhc_boundary::LaneCast>().read_unaligned()))),
            "flash_hybrid_copy"=>Some((Kind::Copy,encode::copy(args[0].cast::<crate::kernel_params::HybridCopy>().read_unaligned()))),
            "mhc_prepare"=>Some((Kind::Prepare,encode::prepare(args[0].cast::<crate::mhc_params::Prepare>().read_unaligned()))),
            "mhc_bf16_collapse"=>Some((Kind::Collapse,encode::collapse(args[0].cast::<crate::mhc_params::Collapse>().read_unaligned()))),
            "mhc_bf16_combine"=>Some((Kind::Combine,encode::combine(args[0].cast::<crate::mhc_params::Combine>().read_unaligned()))),
            "model_rms_partials"=>Some((Kind::RmsPartials,encode::rms(args[0].cast::<crate::kernel_params::ModelRms>().read_unaligned(),Kind::RmsPartials))),
            "model_rms_apply"=>Some((Kind::RmsApply,encode::rms(args[0].cast::<crate::kernel_params::ModelRms>().read_unaligned(),Kind::RmsApply))),
            "model_elementwise"=>Some((Kind::Elementwise,encode::elementwise(args[0].cast::<crate::kernel_params::ModelElementwise>().read_unaligned()))),
            "project_f32_ordered"=>{if args.len()!=5 {return Err("ordered projection argument count differs".into());}let mut c=Command::new(Kind::Plain);for i in 0..3 {c.word(i,args[i].cast::<u64>().read_unaligned());}for i in 3..5 {c.word(i,args[i].cast::<u32>().read_unaligned() as u64);}Some((Kind::Plain,c))},
            "tp_mailbox_put_finite"|"tp_mailbox_take"=>{if args.len()!=10 {return Err("resident mailbox argument count differs".into());}let kind=if name=="tp_mailbox_take" {Kind::Take} else {Kind::Put};let mut c=Command::new(kind);for (i,p) in args.iter().enumerate() {c.word(i,p.cast::<u64>().read_unaligned());}Some((kind,c))},
            _=>None,
        }};
        if let Some((kind,command))=command {self.append_resident(kind,command,grid)?;Ok(true)} else {self.flush_resident_launches()?;Ok(false)}
    }
    pub(super) fn record_resident_copy(&self,source:u64,target:u64,words:usize)->Result<bool,String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        if !self.resident_launches_active() {return Ok(false);}
        if words==0 {return Ok(true);}
        let bytes=byte_count(words,4)? as u64;
        if source!=0 && source!=target && source<target+bytes && target<source+bytes {self.flush_resident_launches()?;return Ok(false);}
        let kind=if source==0 {Kind::ZeroWords} else {Kind::CopyWords};let mut c=Command::new(kind);c.word(0,source);c.word(1,target);c.word(2,words as u64);
        self.append_resident(kind,c,[(words as u32).div_ceil(128),1,1,128,0])?;Ok(true)
    }
    pub(super) fn flush_resident_launches(&self)->Result<(),String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        let calls={let mut r=self.recorder.borrow_mut();if r.paused || r.calls.is_empty() {return Ok(());}r.paused=true;std::mem::take(&mut r.calls)};
        // SAFETY: Each captured node uses its original validated ABI and grid.
        // The private table stays aligned and owned with its executable; uploads
        // and replay share one stream. Queue retention covers every indirect
        // operand through submission and the eventual real completion, including
        // the capture constructor drain. The table contains initialized fields.
        let result=unsafe {(|| {
            let key=calls.iter().map(|c|c.grid).collect::<Vec<_>>();
            if !self.recorder.borrow().pieces.contains_key(&key) {
                if self.recorder.borrow().pieces.len()>=128 {self.synchronize()?;self.recorder.borrow_mut().pieces.clear();}
                let table=self.vector(calls.len()*core::mem::size_of::<Command>()/4)?;
                // SAFETY: The table is private aligned persistent storage. Each
                // replay node names one exact command index and the original
                // validated grid. Capture allocates no storage or reads results.
                // All indirect operands remain retained while unsubmitted is
                // true, including the capture constructor's real stream drain.
                let graph=self.capture_batch_bounded(calls.len(),|| {
                    for (index,call) in calls.iter().enumerate() {let g=call.grid;let (mut address,mut index)=(table.device_address(),index as u64);
                        self.launch_flash_grid(g.kind.kernel(),g.blocks,g.rows,g.planes,g.threads,g.shared,&mut [parameter(&mut address),parameter(&mut index)])?;
                    }Ok(())
                })?;
                self.recorder.borrow_mut().pieces.insert(key.clone(),Piece {graph,table});
            }
            let piece=self.recorder.borrow_mut().pieces.remove(&key).ok_or("resident piece disappeared")?;
            let mut bytes=Vec::with_capacity(calls.len()*core::mem::size_of::<Command>());
            for call in &calls {bytes.extend_from_slice(&call.command.kind.to_ne_bytes());for word in call.command.reserved {bytes.extend_from_slice(&word.to_ne_bytes());}bytes.extend_from_slice(&call.command.payload);}
            // write_buffer's immutable pinned staging stays retained until the
            // actual drain, and upload/replay/next upload share one stream.
            self.write_buffer(&piece.table.buffer,0,&bytes)?;
            self.context.pending.set(true);
            // SAFETY: The executable owns only its persistent table address.
            // All table-referenced operands are held by queue deferred frees
            // until this replay has actually completed. Table replacement is
            // ordered after prior replay. Captured grids and ABI are unchanged.
            let result=self.replay_decode(&piece.graph);
            self.recorder.borrow_mut().pieces.insert(key,piece);result
        })()};
        self.context.resident_unsubmitted.set(false);self.recorder.borrow_mut().paused=false;
        if let Err(error)=result {
            // Accepted producers cannot be retried after a partial submission.
            // Drain earlier work before retiring the queue and its dependent values.
            let _drained=self.synchronize();self.recorder.borrow_mut().failure=Some(error.clone());Err(error)
        } else {Ok(())}
    }
}
