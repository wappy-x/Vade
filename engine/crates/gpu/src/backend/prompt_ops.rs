//! Layer-major prompt rows on the ordered stream. Readback waits for completion.
use super::*;
use crate::prompt_ops::{Op,Shape,Arguments};
impl<'a> Backend<'a> {
    /// Enqueue one prompt operation. Later consumers on this stream follow it.
    /// Dropping an owner drains the stream first, so borrowed inputs stay live.
    #[allow(clippy::if_same_then_else)] // the branches are separate test cases.
    pub fn prompt_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;5],weight:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.prompt_operation_output(shape,input,weight,None)
    }
    /// An optional private destination view. It must not overlap any input. The
    /// ordered stream owns its writes until completion, including mapped storage.
    #[allow(clippy::if_same_then_else)] // The launch branches are separate device cases.
    pub fn prompt_operation_output(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;5],weight:Option<&DeviceVector<'_>>,destination:Option<DeviceVector<'a>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let (lengths,weights,units,words)=shape.lengths().ok_or("invalid prompt row shape")?;
        // The Inkling block arms run in their own kernel entry (Ruling R6g), never in the general prompt rows.
        if shape.op==crate::prompt_ops::Op::Wrapped && crate::block_ops::is_block(shape.inner.op) {return Err("a block arm runs in its own entry, not in prompt rows".into());}
        // No prompt kernel reads the round mark of a fold (`causal_ops::FOLD_ROUNDS`): a wrapped fold is the fold alone.
        if shape.op==crate::prompt_ops::Op::Wrapped && shape.inner.op==crate::causal_ops::Op::ExpertFold && shape.inner.dim==crate::causal_ops::FOLD_ROUNDS {return Err("a wrapped fold does not round its rows".into());}
        let mut addresses=[4u64;5];
        for (i,(vector,count)) in input.into_iter().zip(lengths).enumerate() {
            match vector {Some(v) if count>0=>{self.check_vector_prefix(v,count)?;
                if v.is_segmented() {return Err("prompt row input must be contiguous".into());}addresses[i]=v.device_address();},
                _ if count==0=>{},_=>return Err("prompt row input count differs".into())}
        }
        let address=match weight {Some(w) if weights>0=>{self.check_vector(w,weights)?;
            if w.is_segmented() {return Err("prompt row weight must be contiguous".into());}w.device_address()},
            _ if weights==0=>4,_=>return Err("prompt row weight count differs".into())};
        let total=units.checked_mul(words).ok_or("prompt row output overflow")?;
        let mut output=match destination {Some(out)=>{
            self.check_vector(&out,total.max(1))?;if out.is_segmented() {return Err("prompt destination must be contiguous".into());}
            let end=out.device_address().checked_add((total*4) as u64).ok_or("prompt destination overflow")?;
            if addresses.into_iter().zip(lengths).any(|(start,n)|n>0 && start<end && start+(n*4) as u64>out.device_address()) {
                return Err("prompt destination overlaps an input".into());
            }
            if weights>0 && address<end && address+(weights*4) as u64>out.device_address() {return Err("prompt destination overlaps its weight".into());}
            out},None=>self.vector(total.max(1))?};let mut filled=false;
        // A group past the swept token range of a reading kernel runs that kernel over runs of tokens (prompt_chunks): each run
        // reads its own tokens' words and writes its own tokens' output words, so the bits are those of the one launch.
        // An entry: shape, input addresses and lengths, units, words a unit, and the address of its output words inside `output`.
        let base=output.device_address();let runs=crate::prompt_chunks::runs(shape);let chunked=runs.is_some();
        let mut launches=Vec::new();
        match runs {
            Some(runs)=>for run in runs {
                let (run_lengths,run_weights,run_units,run_words)=run.shape.lengths().ok_or("invalid prompt row run")?;
                if run_weights!=weights {return Err("prompt row run weight extent differs".into());}
                let mut run_addresses=addresses;
                for i in 0..5 {if run_lengths[i]>0 {run_addresses[i]+=4*run.inputs[i] as u64;}}
                launches.push((run.shape,run_addresses,run_lengths,run_units,run_words,base+4*run.output as u64));
            },
            None=>launches.push((shape,addresses,lengths,units,words,base)),
        }
        // One launch for each entry, in this function (its one unsafe block). A kernel that needs a zeroed output zeroes the
        // whole of `output` once (`filled`), before the first launch.
        for (shape,addresses,lengths,units,words,out) in launches {
            if chunked {crate::prompt_launches::record("prompt_token_run");}
            // Plain rows and mapped scores run in shared-memory tiles when shape and alignment qualify, same bytes.
            let (rows,tokens,heads)=(shape.rows as usize,shape.tokens as usize,shape.heads as usize);
            // A draft verify pass has 1 to 16 tokens. The small-count kernels run a warp for each unit instead of one thread and go first
            // (small_choice.rs: their order is tested on the host); then lane K's and lane R's tiles; then prompt_rows.
            // The latent tile: its third form where the kernels take the launch (latent_absorb_rows::third: the same bits), else the first.
            let third=crate::latent_absorb_rows::third(shape,addresses,address,out);
            if crate::latent_absorb_rows::packed(shape.op) && third.is_none() {return Err("packed latent cache shape or alignment is not admitted".into());}
            let small=third.or_else(||crate::latent_absorb_rows::launch(shape)).or_else(||crate::small_choice::prompt(&shape,addresses[0],address));
            let tile=if small.is_some() {small} else {match shape.op {
                Op::ProjectRows if crate::plain_tile::narrow_admitted(rows,tokens,shape.cols as usize,addresses[0],address)=>Some(("plain_rows_narrow",crate::plain_tile::narrow_blocks(rows,tokens).ok_or("plain narrow grid overflow")?,crate::plain_tile::NARROW_THREADS)),
                Op::ProjectRows if crate::plain_tile::admitted(rows,tokens,shape.cols as usize,addresses[0],address)=>Some(("plain_rows_tile",crate::plain_tile::blocks(rows,tokens).ok_or("plain tile grid overflow")?,crate::plain_tile::THREADS)),
                Op::ScoresMap if crate::scores_tile::admitted(rows,tokens,heads,shape.dim as usize,shape.taps as usize,addresses[0],addresses[1])=>Some(("scores_rows_tile",crate::scores_tile::blocks(rows,tokens,heads).ok_or("scores tile grid overflow")?,crate::scores_tile::THREADS)),
                Op::ScoresMapPacked if crate::scores_tile::packed_admitted(rows,tokens,heads,shape.dim as usize,shape.taps as usize,addresses[0],addresses[1])=>Some(("scores_rows_tile_packed",crate::scores_tile::blocks(rows,tokens,heads).ok_or("scores tile grid overflow")?,crate::scores_tile::THREADS)),
                // The indexer's pool scores of 1 to 16 token rows (a verify pass, a step of several chats): a warp a (token, pool) word, a lane a head (pool_scores_small.rs).
                Op::PoolScoresRows if crate::pool_scores_small::admitted(rows,tokens,heads,shape.dim as usize,shape.taps as usize)=>Some(("prompt_pool_scores",crate::pool_scores_small::blocks(rows,tokens).ok_or("pool scores grid overflow")?,crate::pool_scores_small::THREADS)),
                // The indexer's pool scores: 64 pools x 64 tokens a block, every head's chain as the unit makes it (pool_scores_tile.rs).
                Op::PoolScoresRows if crate::pool_scores_tile::admitted(rows,tokens,heads,shape.dim as usize,shape.taps as usize,addresses[0],addresses[1])=>Some(("pool_scores_tile",crate::pool_scores_tile::blocks(rows,tokens).ok_or("pool scores tile grid overflow")?,crate::pool_scores_tile::THREADS)),
                // The recurrent update of a prompt group (17 to 1,024 tokens): the grid of step_rows_block, the tokens staged ahead (step_rows_tile.rs).
                Op::StepRows if crate::step_tile::admitted(tokens,heads,shape.dim as usize)=>Some(("step_rows_tile",crate::step_tile::blocks(heads,shape.dim as usize).ok_or("step tile grid overflow")?,crate::step_tile::THREADS)),
                // One block per head and 32 values, one thread per value.
                Op::StepRows if shape.dim.is_multiple_of(32) && (32..=128).contains(&shape.dim)=>Some(("step_rows_block",shape.heads*(shape.dim/32),32)),
                _=>None}};
            let selecting=crate::select_admission::admitted(shape);
            // The two data-movement kernels write every output word once (prompt_move.rs), so no zero fill precedes them.
            let permuting=!selecting && tile.is_none() && crate::prompt_admission::permute_admitted(shape,addresses[0],out);
            let placing=!selecting && tile.is_none() && crate::prompt_admission::place_admitted(shape);
            // A matrix of whole 4 x 4 tiles transposes through a 32 x 32 shared-memory tile (transpose_smem.rs of transpose_tiles.rs): same bits, coalesced both ways.
            let transposing=!selecting && tile.is_none() && crate::prompt_admission::transpose_admitted(shape,addresses[0],out);
            // Wrapped Prepare, GatedNorm, Route and ExpertFold, and the convolution rows, run as warps or register threads that write every word.
            let fast=if selecting || tile.is_some() || permuting || placing || transposing {None} else if crate::softmax_rows::admitted(shape) {Some("prompt_softmax_rows")} else {crate::prompt_admission::fast_kernel(shape)};
            self.require_arm_kernels(&[if selecting {"prompt_select_rows"} else if permuting {"prompt_permute"} else if placing {"prompt_place"} else if transposing {"transpose_smem"} else if let Some(name)=fast {name} else {tile.map_or("prompt_rows",|t|t.0)}])?;
            // A recorded zero fill would leave as a piece of its own before a direct row kernel, one table
            // upload and one graph launch each. So earlier records go first and the fill is a direct clear.
            // Kernels that write every output word from any initial content (prompt_writes, device-swept from a poisoned output) run unzeroed.
            let fills=!permuting && !placing && !transposing && fast.is_none() && !crate::latent_absorb_rows::admitted(shape.op);
            // transpose_smem writes every output word (swept from a poisoned output in kernel-bench --transpose-smem): no fill, counted like the others.
            let recurrent_writes=tokens>1 && (tile.is_some_and(|t|t.0=="prompt_rows_small")
                || shape.op==Op::HistoryRows && units>0);
            if transposing {crate::prompt_launches::record("prompt_fill_skipped");}
            else if fills && (recurrent_writes || crate::prompt_writes::writes_every_word(shape.op==Op::Transpose,shape.rows,shape.cols,tile.map(|t|t.0))) {crate::prompt_launches::record("prompt_fill_skipped");}
            else if fills && !filled {
                // The fill clears the whole output, so it must come before the first run writes. Runs of one operation take one kernel
                // family (all fill or none do); a later run that asked for a fill would clear the runs before it: refuse, never corrupt.
                if out!=output.device_address() {return Err("prompt row run needs a zero fill after an earlier run wrote".into());}
                if self.resident_rows_active() {self.zero_vector(&mut output)?;} else {self.unrecorded(||self.zero_vector(&mut output))?;}filled=true;
            }
            if units==0 {continue;}
            let threads=shape.threads(units,(addresses[0]|addresses[1])&15==0).ok_or("prompt row thread overflow")?;
            let mut args=Arguments {input:addresses,lengths:lengths.map(|n|n as u64),weight:address,weights:weights as u64,
                output:out,units:threads as u64,words:words as u64,shape};
            // A transpose of whole tiles copies 16 words per thread, so it needs a sixteenth of the threads.
            let tiles=!transposing && shape.op==Op::Transpose && shape.rows.is_multiple_of(4) && shape.cols.is_multiple_of(4) && (addresses[0]|out)&15==0;
            if tiles {self.require_arm_kernels(&["transpose_tiles"])?;args.units=(threads/16) as u64;}
            let (kernel,blocks,rows,group)=if placing {let (gx,gy)=crate::prompt_admission::place_grid(shape);("prompt_place",gx,gy,crate::prompt_admission::MOVE_THREADS)}
                else if transposing {let (gx,gy)=crate::prompt_admission::transpose_grid(shape);("transpose_smem",gx,gy,crate::prompt_admission::TRANSPOSE_THREADS)}
                else if permuting {("prompt_permute",crate::prompt_admission::permute_blocks(shape),1,crate::prompt_admission::MOVE_THREADS)}
                else if let Some(name)=fast {(name,if name=="prompt_softmax_rows" {shape.tokens*shape.heads} else {crate::prompt_admission::fast_blocks(shape)},1,crate::prompt_admission::FAST_THREADS)}
                else if selecting {("prompt_select_rows",shape.tokens,1,crate::select_admission::THREADS)}
                else {match tile {Some(t)=>(t.0,t.1,1,t.2),None if tiles=>("transpose_tiles",u32::try_from((threads/16).div_ceil(128)).map_err(|_|"prompt row grid overflow")?,1,128),None=>("prompt_rows",u32::try_from(threads.div_ceil(128)).map_err(|_|"prompt row grid overflow")?,1,128)}};
            // A third-form launch counts as the latent tile it replaces (the gate reads that count) and under its own name.
            if third.is_some() {crate::prompt_launches::record("latent_absorption_tile");crate::prompt_launches::record("latent_tile3");} else {crate::prompt_launches::record(kernel);}
            let big=crate::prompt_admission::big_group;
            if (permuting || placing) && big(shape.rows as usize/crate::prompt_admission::ROUTED) || transposing && big(shape.cols as usize) || fast.is_some() && big(tokens) {crate::prompt_launches::record("prompt_big_group");}
            // SAFETY: Shape bounds every output unit and every read indexed by shape, and
            // the kernel receives those exact extents. Counts and row IDs read from a
            // selected input are data: the kernel checks each against its extent and traps,
            // and the launch then fails. check_vector_prefix checks owners and
            // contexts. Inputs are read-only. The private output is disjoint and contiguous. It is zeroed earlier on
            // this ordered stream, except where recurrent_writes or prompt_writes::writes_every_word holds: there the kernel
            // stores every output word through a raw pointer, never reads one and builds no reference over the output, so the fill is skipped. Buffer::drop drains the
            // stream, so all owners outlive the launch. The argument is copied at launch. The
            // Absorbed row kernels write every output word, validate selected counts and IDs, and keep
            // each token's strip order. Their launch geometry and extents come from latent_absorb_rows.
            // tile and small kernels take the same arguments, with their own block sizes and the alignment
            // their admitted function checked above. The scores kernel checks each map entry.
            // The step kernel needs the block and thread counts chosen above and rechecks them. The data-movement and wrapped-operation
            // kernels take the same arguments and write every output word once (prompt_admission), so they run on the unzeroed output;
            // prompt_softmax_rows launches exactly tokens*heads blocks, one per token and head; each writes every word
            // of its row: zeros at count..cols and probabilities at 0..count. The place kernel's second grid axis is its channel tiles.
            unsafe {self.launch_flash_rows(kernel,blocks,rows,group,0,&mut [parameter(&mut args)])}?;
        }
        Ok(output)
    }
    /// Run one recurrent layer over consecutive rows of one chat. All outputs
    /// have private storage and are completely written on the ordered stream.
    pub fn recurrent_rows(&self,shape:crate::recurrent_rows::Shape,input:[&DeviceVector<'_>;7],
        gate:&DeviceVector<'_>,weight:[&DeviceVector<'_>;3],keep:bool)
        ->Result<([DeviceVector<'a>;3],Option<[DeviceVector<'a>;2]>),String> {
        use crate::recurrent_rows::{Arguments,THREADS};
        self.require_resident_stream()?;
        let (inputs,weights,[history,state,rows,prepared,decay])=shape.lengths(keep).ok_or("invalid recurrent row shape")?;
        for (source,words) in input.into_iter().zip(inputs).chain(weight.into_iter().zip(weights)).chain([(gate,rows)]) {
            self.check_vector(source,words)?;
            if source.is_segmented() {return Err("recurrent row input must be contiguous".into());}
        }
        self.require_arm_kernels(&["recurrent_rows"])?;
        let next=[self.vector(history)?,self.vector(state)?,self.vector(rows)?];
        let saved=if keep {Some([self.vector(prepared)?,self.vector(decay)?])} else {None};
        let [prepared,decay]=saved.as_ref().map_or([0;2],|pair|[pair[0].device_address(),pair[1].device_address()]);
        let mut args=Arguments {shape,inputs:input.map(DeviceVector::device_address),gate:gate.device_address(),
            weights:weight.map(DeviceVector::device_address),
            outputs:[next[0].device_address(),next[1].device_address(),next[2].device_address(),prepared,decay]};
        crate::prompt_launches::record("recurrent_rows");
        // SAFETY: Exact input, gate and weight extents, context and contiguity
        // were checked above. Fresh owners do not alias inputs or each other.
        // The head count and block size give every output word one writer, and
        // no thread reads uninitialized output. The argument is copied at
        // launch. The queue keeps all owners through the ordered stream drain.
        unsafe {self.launch_flash("recurrent_rows",shape.heads,THREADS,&mut [parameter(&mut args)])}?;
        Ok((next,saved))
    }
    /// Busy-poll the stream instead of the driver's default wait, whose wake-up
    /// latency dominates short GPU segments between host collectives.
    pub fn set_spin_waits(&self, spin: bool) { self.spin_waits.set(spin); }
    /// Free the spare vectors a large group left. They fill the pool at sizes decode
    /// never asks for, so decode's own vectors would miss it on every token.
    /// Start a new high mark for one prompt group. Returns the whole run's peak for `end_group_peak`.
    pub fn begin_group_peak(&self)->usize {let whole=self.context.peak_accounted.get();self.context.peak_accounted.set(self.context.accounted_bytes());whole}
    /// The group's peak. The whole run's peak stays true, because the two are joined here.
    pub fn end_group_peak(&self,whole:usize)->usize {let group=self.context.peak_accounted.get();self.context.peak_accounted.set(whole.max(group));group}
    pub fn release_group_spares(&self)->Result<(),String> {self.synchronize()?;self.context.clear_vector_pool();Ok(())}
    /// Drain pending returns before recording the vector pool's spare capacities.
    pub fn begin_prompt_cohort(&self)->Result<(),String> {
        self.synchronize()?;
        self.context.vector_pool.borrow_mut().begin_prompt_cohort()
    }
    /// Return wide scratch and drain dropped values while the cohort owns their cost.
    pub fn prompt_cohort_chat(&self, chat: Option<usize>) {self.context.vector_pool.borrow_mut().prompt_cohort_chat(chat);}
    pub fn finish_prompt_cohort(&self)->Result<(usize,usize,usize),String> {
        self.nv_tc_scratch_return();
        // A failed drain still ends the ledger, so a later cohort can begin and no later return is counted for this one.
        if let Err(error) = self.synchronize() {self.context.vector_pool.borrow_mut().abandon_prompt_cohort();return Err(error);}
        self.context.vector_pool.borrow_mut().finish_prompt_cohort(self.context)
    }
    /// After a prompt group: drain, then trim the vector pool to leave `room` under its cap,
    /// as each decode token does, instead of releasing every spare. The next group (or the
    /// next session's first group) takes its own buffer sizes back from the pool.
    pub fn keep_group_spares(&self,room:usize)->Result<(),String> {self.synchronize()?;self.context.trim_vector_pool(room);Ok(())}
    pub fn prompt_spare_bytes(&self) -> usize { self.context.vector_pool.borrow().spare_bytes() }
    pub fn return_prompt_spares(&self, target: usize) -> Result<(usize, usize), String> {
        self.synchronize()?;
        let mut pool = self.context.vector_pool.borrow_mut();
        let before = pool.spare_bytes();
        pool.return_vector_spares(self.context, target);
        Ok((before, pool.spare_bytes()))
    }
    /// LEASE SPARES: after a prompt group whose prompt goes on, when its lease goes back: drain, then end the group's pool
    /// cap and ceiling (`group_vector_spares`) without freeing a spare. The next group takes the same buffer sizes back
    /// from the pool. Free storage only: the device budget and the host pressure rule still release spares.
    pub fn end_group_spares(&self)->Result<(),String> {self.synchronize()?;self.context.end_group_vector_pool();Ok(())}
    /// A decode token starts: release the spare vectors kept longest until `room` bytes fit
    /// under the pool's cap. Buffers whose size follows the context are never asked for
    /// again; without this they fill the pool and every later buffer is allocated anew.
    pub fn trim_vector_spares(&self,room:usize) {self.context.trim_vector_pool(room);}
    /// A prompt group of more rows than the pool's least cap was set for starts: until the next trim or
    /// release of the pool's spares its byte cap is `cap` (when that is above the pool's own), and a return
    /// that finds the pool full stays while the spares kept longest leave. Free storage only: no value
    /// changes, the device budget and the host pressure rule release spares as before, and
    /// `keep_group_spares` at the group's end leaves the same bytes as before. `ceiling`: the device bytes
    /// admission charged for the work now live (usize::MAX: not known); the spares this cap adds leave before
    /// an allocation takes the device past it.
    pub fn group_vector_spares(&self,cap:usize,ceiling:usize) {self.context.group_vector_pool(cap,ceiling);}
    /// A decode step starts (`true`) or prompt work starts (`false`). During a decode step pooled vectors come from
    /// the decode reserve first and return to it whenever they are freed; prompt work never takes from it.
    pub fn decode_phase(&self,on:bool) {self.context.decoding.set(on);}
    pub fn decoding(&self)->bool {self.context.decoding.get()}
    /// Byte cap of the decode reserve; 0 turns it off. See `decode_reserve`.
    pub fn configure_decode_reserve(&self,cap:usize) {if cap==0 {self.context.clear_decode_reserve();}self.context.decode_reserve.borrow_mut().configure(cap);}
    pub fn decode_reserve_configured(&self)->bool {self.context.decode_reserve.borrow().configured()}
    /// Spare bytes of the decode reserve now, the most it held, and its cap.
    pub fn decode_reserve_bound(&self)->(usize,usize,usize) {self.context.decode_reserve.borrow().bound()}
    /// Screens only: divide the vector pool's cap and every trim room by 2^shift, so a small fixture meets the
    /// bounds a full model meets. The decode reserve's cap is the caller's own number.
    pub fn scale_vector_pool(&self,shift:u32) {self.context.vector_pool.borrow_mut().scale(shift);}
    /// Counts of the vector pool since the load: new allocations, reuses, returns and releases.
    pub fn vector_pool_stats(&self)->crate::KvArenaStats {self.context.vector_pool_stats()}
    /// Observe free storage left by decode-start trims. These bytes use the
    /// separate spare allowance in service_memory::workspace, not live scratch.
    pub fn reset_vector_pool_bound(&self) {self.context.vector_pool.borrow_mut().reset_trim_bound();}
    /// Trim samples, largest free-spare byte count after a trim, and pool cap.
    pub fn vector_pool_bound(&self)->(usize,usize,usize) {self.context.vector_pool.borrow().trim_bound()}
}
