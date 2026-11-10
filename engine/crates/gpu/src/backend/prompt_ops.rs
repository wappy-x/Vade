//! Layer-major prompt rows on the ordered stream. Readback waits for completion.
use super::*;
use crate::prompt_ops::{Op,Shape,Arguments};
impl<'a> Backend<'a> {
    /// Enqueue one prompt operation. Later consumers on this stream follow it.
    /// Dropping an owner drains the stream first, so borrowed inputs stay live.
    #[allow(clippy::if_same_then_else)] // the branches are separate test cases.
    pub fn prompt_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;5],weight:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let (lengths,weights,units,words)=shape.lengths().ok_or("invalid prompt row shape")?;
        let mut addresses=[4u64;5];
        for (i,(vector,count)) in input.into_iter().zip(lengths).enumerate() {
            match vector {Some(v) if count>0=>{self.check_vector_prefix(v,count)?;
                if v.is_segmented() {return Err("prompt row input must be contiguous".into());}addresses[i]=v.device_address();},
                _ if count==0=>{},_=>return Err("prompt row input count differs".into())}
        }
        let address=match weight {Some(w) if weights>0=>{self.check_vector(w,weights)?;
            if w.is_segmented() {return Err("prompt row weight must be contiguous".into());}w.device_address()},
            _ if weights==0=>4,_=>return Err("prompt row weight count differs".into())};
        // Plain rows and mapped scores run in shared-memory tiles when shape and alignment qualify, same bytes.
        let (rows,tokens,heads)=(shape.rows as usize,shape.tokens as usize,shape.heads as usize);
        // A draft verify pass has 1 to 16 tokens. The small-count kernels run a warp for each unit instead of one thread and go first
        // (small_choice.rs: their order is tested on the host); then lane K's and lane R's tiles; then prompt_rows.
        let small=crate::small_choice::prompt(&shape,addresses[0],address);
        let tile=if small.is_some() {small} else {match shape.op {
            Op::ProjectRows if crate::plain_tile::narrow_admitted(rows,tokens,shape.cols as usize,addresses[0],address)=>Some(("plain_rows_narrow",crate::plain_tile::narrow_blocks(rows,tokens).ok_or("plain narrow grid overflow")?,crate::plain_tile::NARROW_THREADS)),
            Op::ProjectRows if crate::plain_tile::admitted(rows,tokens,shape.cols as usize,addresses[0],address)=>Some(("plain_rows_tile",crate::plain_tile::blocks(rows,tokens).ok_or("plain tile grid overflow")?,crate::plain_tile::THREADS)),
            Op::ScoresMap if crate::scores_tile::admitted(rows,tokens,heads,shape.dim as usize,shape.taps as usize,addresses[0],addresses[1])=>Some(("scores_rows_tile",crate::scores_tile::blocks(rows,tokens,heads).ok_or("scores tile grid overflow")?,crate::scores_tile::THREADS)),
            Op::ScoresMapPacked if crate::scores_tile::packed_admitted(rows,tokens,heads,shape.dim as usize,shape.taps as usize,addresses[0],addresses[1])=>Some(("scores_rows_tile_packed",crate::scores_tile::blocks(rows,tokens,heads).ok_or("scores tile grid overflow")?,crate::scores_tile::THREADS)),
            // One block per head and 32 values, one thread per value.
            Op::StepRows if shape.dim.is_multiple_of(32) && (32..=128).contains(&shape.dim)=>Some(("step_rows_block",shape.heads*(shape.dim/32),32)),
            _=>None}};
        let selecting=crate::select_admission::admitted(shape);
        let total=units.checked_mul(words).ok_or("prompt row output overflow")?;
        let mut output=self.vector(total.max(1))?;
        // The two data-movement kernels write every output word once (prompt_move.rs), so no zero fill precedes them.
        let permuting=!selecting && tile.is_none() && crate::prompt_admission::permute_admitted(shape,addresses[0],output.device_address());
        let placing=!selecting && tile.is_none() && crate::prompt_admission::place_admitted(shape);
        // A matrix of whole 4 x 4 tiles transposes through a 32 x 32 shared-memory tile (transpose_smem.rs of transpose_tiles.rs): same bits, coalesced both ways.
        let transposing=!selecting && tile.is_none() && crate::prompt_admission::transpose_admitted(shape,addresses[0],output.device_address());
        // Wrapped Prepare, GatedNorm, Route and ExpertFold, and the convolution rows, run as warps or register threads that write every word.
        let fast=if selecting || tile.is_some() || permuting || placing || transposing {None} else if crate::softmax_rows::admitted(shape) {Some("prompt_softmax_rows")} else {crate::prompt_admission::fast_kernel(shape)};
        self.require_arm_kernels(&[if selecting {"prompt_select_rows"} else if permuting {"prompt_permute"} else if placing {"prompt_place"} else if transposing {"transpose_smem"} else if let Some(name)=fast {name} else {tile.map_or("prompt_rows",|t|t.0)}])?;
        // A recorded zero fill would leave as a piece of its own before a direct row kernel, one table
        // upload and one graph launch each. So earlier records go first and the fill is a direct clear.
        // Kernels that write every output word from any initial content (prompt_writes, device-swept from a poisoned output) run unzeroed.
        let fills=!permuting && !placing && !transposing && fast.is_none();
        // transpose_smem writes every output word (swept from a poisoned output in kernel-bench --transpose-smem): no fill, counted like the others.
        if transposing {crate::prompt_launches::record("prompt_fill_skipped");}
        else if fills && crate::prompt_writes::writes_every_word(shape.op==Op::Transpose,shape.rows,shape.cols,tile.map(|t|t.0)) {crate::prompt_launches::record("prompt_fill_skipped");}
        else if fills {self.unrecorded(||self.zero_vector(&mut output))?;}
        if units==0 {return Ok(output);}
        let threads=shape.threads(units,(addresses[0]|addresses[1])&15==0).ok_or("prompt row thread overflow")?;
        let mut args=Arguments {input:addresses,lengths:lengths.map(|n|n as u64),weight:address,weights:weights as u64,
            output:output.device_address(),units:threads as u64,words:words as u64,shape};
        // A transpose of whole tiles copies 16 words per thread, so it needs a sixteenth of the threads.
        let tiles=!transposing && shape.op==Op::Transpose && shape.rows.is_multiple_of(4) && shape.cols.is_multiple_of(4) && (addresses[0]|output.device_address())&15==0;
        if tiles {self.require_arm_kernels(&["transpose_tiles"])?;args.units=(threads/16) as u64;}
        let (kernel,blocks,rows,group)=if placing {let (gx,gy)=crate::prompt_admission::place_grid(shape);("prompt_place",gx,gy,crate::prompt_admission::MOVE_THREADS)}
            else if transposing {let (gx,gy)=crate::prompt_admission::transpose_grid(shape);("transpose_smem",gx,gy,crate::prompt_admission::TRANSPOSE_THREADS)}
            else if permuting {("prompt_permute",crate::prompt_admission::permute_blocks(shape),1,crate::prompt_admission::MOVE_THREADS)}
            else if let Some(name)=fast {(name,if name=="prompt_softmax_rows" {shape.tokens*shape.heads} else {crate::prompt_admission::fast_blocks(shape)},1,crate::prompt_admission::FAST_THREADS)}
            else if selecting {("prompt_select_rows",shape.tokens,1,crate::select_admission::THREADS)}
            else {match tile {Some(t)=>(t.0,t.1,1,t.2),None if tiles=>("transpose_tiles",u32::try_from((threads/16).div_ceil(128)).map_err(|_|"prompt row grid overflow")?,1,128),None=>("prompt_rows",u32::try_from(threads.div_ceil(128)).map_err(|_|"prompt row grid overflow")?,1,128)}};
        crate::prompt_launches::record(kernel);
        // SAFETY: Shape bounds every output unit and every read indexed by shape, and
        // the kernel receives those exact extents. Counts and row IDs read from a
        // selected input are data: the kernel checks each against its extent and traps,
        // and the launch then fails. check_vector_prefix checks owners and
        // contexts. Inputs are read-only. The private output is disjoint and contiguous. It is zeroed earlier on
        // this ordered stream, except where prompt_writes::writes_every_word holds (a Transpose of a swept shape, and plain_rows_small): there the kernel
        // stores every output word through a raw pointer, never reads one and builds no reference over the output, so the fill is skipped. Buffer::drop drains the
        // stream, so all owners outlive the launch. The argument is copied at launch. The
        // tile and small kernels take the same arguments, with their own block sizes and the alignment
        // their admitted function checked above. The scores kernel checks each map entry.
        // The step kernel needs the block and thread counts chosen above and rechecks them. The data-movement and wrapped-operation
        // kernels take the same arguments and write every output word once (prompt_admission), so they run on the unzeroed output;
        // prompt_softmax_rows launches exactly tokens*heads blocks, one per token and head; each writes every word
        // of its row: zeros at count..cols and probabilities at 0..count. The place kernel's second grid axis is its channel tiles.
        unsafe {self.launch_flash_rows(kernel,blocks,rows,group,0,&mut [parameter(&mut args)])}?;
        Ok(output)
    }
    /// Free the spare vectors a large group left. They fill the pool at sizes decode
    /// never asks for, so decode's own vectors would miss it on every token.
    /// Start a new high mark for one prompt group. Returns the whole run's peak for `end_group_peak`.
    pub fn begin_group_peak(&self)->usize {let whole=self.context.peak_accounted.get();self.context.peak_accounted.set(self.context.accounted_bytes());whole}
    /// The group's peak. The whole run's peak stays true, because the two are joined here.
    pub fn end_group_peak(&self,whole:usize)->usize {let group=self.context.peak_accounted.get();self.context.peak_accounted.set(whole.max(group));group}
    pub fn release_group_spares(&self)->Result<(),String> {self.synchronize()?;self.context.clear_vector_pool();Ok(())}
    /// After a prompt group: drain, then trim the vector pool to leave `room` under its cap,
    /// as each decode token does, instead of releasing every spare. The next group (or the
    /// next session's first group) takes its own buffer sizes back from the pool.
    pub fn keep_group_spares(&self,room:usize)->Result<(),String> {self.synchronize()?;self.context.trim_vector_pool(room);Ok(())}
    /// A decode token starts: release the spare vectors kept longest until `room` bytes fit
    /// under the pool's cap. Buffers whose size follows the context are never asked for
    /// again; without this they fill the pool and every later buffer is allocated anew.
    pub fn trim_vector_spares(&self,room:usize) {self.context.trim_vector_pool(room);}
    /// Counts of the vector pool since the load: new allocations, reuses, returns and releases.
    pub fn vector_pool_stats(&self)->crate::KvArenaStats {self.context.vector_pool_stats()}
}
