//! The tensor-core kernel family: the second image (`sm_121a`, crate `spark-tc-kernels`), loaded when one of its
//! entries is first launched. The routed-expert NVFP4 products of a prompt group run here on contract (b)
//! (`tc_contract`): `tc_nv_pack` turns the quantizer's FP32 activation rows into packed operands, then `fp4_cohort`
//! runs the block-scaled tensor-core instruction, one launch a layer. A dense prompt group skips the quantizer's rows:
//! `nv_quantize_pack` (the production image) writes the same packed operands straight from the FP32 rows, once for all
//! the matrices that read them (`Backend::expert_group_quantized`).
use super::*;
use crate::expert_ops::{GroupEntry,Op};
use crate::tc_abi::{NvPack,NvPart,NvQuantizePack,NvRows,NV_PACK_THREADS,NV_QUANTIZE_PACK_THREADS,NV_ROWS_BLOCK_TOKENS,NV_ROWS_MAX_COLS,NV_ROWS_THREADS};

/// Where the packed operands of a product launch come from.
pub(super) enum NvOperands<'v,'d> {
    /// The quantizer's rows (17 FP32 words a block, the stride given): `tc_nv_pack` packs them.
    Rows(&'v DeviceVector<'d>,usize),
    /// FP32 activation rows and the input scale of each row: `nv_quantize_pack` quantizes and packs them in one pass.
    Fp32(&'v DeviceVector<'d>,&'v [f32]),
    /// The kept scratch already holds this group's operands: an earlier list of the same call wrote them.
    Kept,
}

/// Entries that live in the second image.
pub(crate) fn in_tc_image(name:&str)->bool {matches!(name,"fp4_cohort"|"fp4_cohort_t64"|"tc_nv_pack")}
/// The product entry and its tokens a block for a word function: one chain runs in the 128-token block; the quarter
/// chain runs in the 64-token block (its second accumulator set does not fit the 128-token block's registers).
pub(crate) fn nv_rows_entry(chains:u32)->(&'static str,u32) {if chains==4 {("fp4_cohort_t64",64)} else {("fp4_cohort",NV_ROWS_BLOCK_TOKENS)}}

/// True when a group table may run on `fp4_cohort`: every part NvRows on column tiles with 16-column scale blocks,
/// equal rows and cols, whole 128-row tiles, whole 64-column steps in every chain, cumulative first rows, the
/// quantizer's stride, one word a unit, aligned owners.
pub(crate) fn nv_rows_tc_admitted(table:&[GroupEntry],stride:usize,cumulative:bool,words:usize,input:u64,chains:u32)->bool {
    let Some(first)=table.first() else {return false};
    let (rows,cols)=(first.shape.rows as usize,first.shape.cols as usize);
    crate::tc_contract::chain_steps(cols,chains).is_some() && cols<=NV_ROWS_MAX_COLS as usize && rows>0 && rows%crate::expert_ops::NV_TILE_ROWS as usize==0
        && words==1 && cumulative && stride==cols/16*17 && input&3==0
        && table.iter().all(|e|e.shape.op==Op::NvRows && e.shape.block_cols==16 && crate::expert_ops::nv_tiled(&e.shape)
            && e.shape.rows as usize==rows && e.shape.cols as usize==cols && e.shape.tokens>0 && (e.weight|e.scales)&15==0)
}

/// The backend's state for the second kernel image (the tensor-core family).
pub(super) struct TcState<'a> {
    /// The image, loaded when one of its entries is first launched.
    module:std::cell::OnceCell<Module<'a>>,
    /// Check tools only: true = dense NVFP4 groups do not use the tensor-core image (they run as decode jobs or the
    /// generic body, the same words). false = the serving choice.
    off:Cell<bool>,
    /// Kept scratch of the group path: packed activation codes, scale codes, the part table and the input scale of each
    /// row. Grown when a group needs more; a fresh device allocation a call costs more than the pack pass itself.
    scratch:RefCell<Option<[DeviceVector<'a>;4]>>,
    /// Check tools only: true = a dense prompt group does not take `nv_quantize_pack` (it runs the quantizer and
    /// `tc_nv_pack`, the same operand bytes). false = the serving choice.
    fused_off:Cell<bool>,
}
impl TcState<'_> {
    pub(super) fn new()->Self {Self {module:std::cell::OnceCell::new(),off:Cell::new(false),scratch:RefCell::new(None),fused_off:Cell::new(false)}}
}
impl<'a> Backend<'a> {
    pub(super) fn tc_module(&self)->Result<&Module<'a>,String> {
        if let Some(module)=self.tc.module.get() {return Ok(module);}
        let ptx=crate::tc_ptx().ok_or("this build has no tensor-core image: run `cargo kernels --tc` before the host build")?;
        let module=Module::load_second(self.context,ptx)?;
        Ok(self.tc.module.get_or_init(||module))
    }
    /// True when this build carries the tensor-core image.
    pub fn tc_image_present(&self)->bool {crate::tc_ptx().is_some()}
    /// Loads the second image now when this build carries it, so the first dense prompt group after a start does not
    /// pay for its PTX compile (MEASURED about 0.29 s on a pair). The same module and entries: no word changes.
    pub fn tc_preload(&self)->Result<(),String> {if self.tc_image_present() {self.tc_module()?;} Ok(())}
    /// Check tools only: false = an admitted dense group runs as decode jobs or the generic body (the same words),
    /// so a tool can time the tensor-core path against them and compare their words.
    pub fn set_nv_tc_groups(&self,on:bool) {self.tc.off.set(!on);}
    pub(super) fn nv_tc_groups_off(&self)->bool {self.tc.off.get()}
    /// Check tools only: false = a dense prompt group runs the quantizer and `tc_nv_pack` (the same operand bytes and
    /// words), so a tool can time `nv_quantize_pack` against them and compare their words.
    pub fn set_nv_quantize_pack(&self,on:bool) {self.tc.fused_off.set(!on);}
    pub(super) fn nv_quantize_pack_off(&self)->bool {self.tc.fused_off.get()}
    /// True when the ACTIVE primary image has the `nv_quantize_pack` entry. A host can run with an older kernel file
    /// (a second image chosen at a request boundary) that lacks it: such a group keeps the two-step path.
    pub(super) fn nv_quantize_pack_present(&self)->bool {self.require_arm_kernels(&["nv_quantize_pack"]).is_ok()}
    /// Check tools only: fills the kept operand scratch (codes and scale codes) with the byte 0xA5, so a byte that a
    /// later operand launch does not store shows in the product words. No effect when there is no scratch yet.
    pub fn nv_tc_scratch_poison(&self)->Result<(),String> {
        let mut scratch=self.tc.scratch.borrow_mut();let Some(kept)=scratch.as_mut() else {return Ok(());};
        for vector in kept.iter_mut().take(2) {let len=vector.len();let poison=self.upload_vector(&vec![f32::from_bits(0xa5a5_a5a5);len])?;self.unrecorded(||self.copy_prefix(&poison,vector,len))?;self.synchronize()?;}
        Ok(())
    }
    /// Check tools only: the packed operands `nv_quantize_pack` writes for `activations.len()` FP32 rows of `cols` words
    /// (row r at the input scale `activations[r]`): (code bytes, scale bytes), read back. Fresh outputs filled with the
    /// byte 0xA5 before the launch (a byte the kernel does not store stays 0xA5 and shows), one launch, one drain.
    pub fn nv_quantize_pack_bytes(&self,input:&DeviceVector<'_>,cols:usize,activations:&[f32])->Result<(Vec<u8>,Vec<u8>),String> {
        self.require_resident_stream()?;let rows=activations.len();
        if rows==0 || cols==0 || !cols.is_multiple_of(16) || input.is_segmented() || input.len()!=rows.checked_mul(cols).ok_or("quantize-pack input overflow")? || input.device_address()&3!=0 {return Err("quantize-pack check extent differs".into());}
        self.check_vector(input,input.len())?;
        let table=self.upload_vector(activations)?;
        let (mut codes,mut scales)=(self.vector(rows*(cols/8))?,self.vector((rows*(cols/16)).div_ceil(4))?);
        for vector in [&mut codes,&mut scales] {let len=vector.len();let poison=self.upload_vector(&vec![f32::from_bits(0xa5a5_a5a5);len])?;self.unrecorded(||self.copy_prefix(&poison,vector,len))?;self.synchronize()?;}
        let mut args=NvQuantizePack {input:input.device_address(),codes:codes.device_address(),scales:scales.device_address(),activations:table.device_address(),
            rows:u32::try_from(rows).map_err(|_|"quantize-pack rows exceed ABI")?,cols:u32::try_from(cols).map_err(|_|"quantize-pack columns exceed ABI")?};
        let blocks=crate::tc_abi::nv_quantize_pack_blocks(rows,cols).ok_or("quantize-pack grid overflow")?;
        self.require_arm_kernels(&["nv_quantize_pack"])?;
        // The checks above give the kernel its contract: rows * cols input words, rows scale words, initialized private outputs
        // of rows * cols / 2 and rows * cols / 16 bytes (rounded up to words), 4-byte aligned, all alive through the drain below.
        self.unrecorded(||self.launch_checked_expert_payload("nv_quantize_pack",blocks,1,NV_QUANTIZE_PACK_THREADS,0,&mut args))?;
        self.synchronize()?;
        let bytes=|v:&DeviceVector<'_>,n:usize|->Result<Vec<u8>,String> {Ok(self.read_vector_values(v)?.iter().flat_map(|w|w.to_bits().to_le_bytes()).take(n).collect())};
        Ok((bytes(&codes,rows*cols/2)?,bytes(&scales,rows*cols/16)?))
    }
    /// The routed-expert products of one admitted group table into `output` (the group's private output; this
    /// path writes every word of it before any read, so the caller does not fill it):
    /// part e's word (row, token) lands at unit `e.first + row * tokens + token`, as every group kernel writes it.
    /// `rows_in` = the input rows the table covers (the end of the last part).
    /// `operands` says how the packed operands of the `rows_in` input rows reach the kept scratch before the product launch.
    pub(super) fn nv_rows_tc(&self,table:&[GroupEntry],operands:NvOperands<'_,'_>,output:&DeviceVector<'_>,rows_in:usize,chains:u32)->Result<(),String> {
        let (rows,cols)=(table[0].shape.rows as usize,table[0].shape.cols as usize);
        let count=u32::try_from(table.len()).map_err(|_|"tensor-core group exceeds ABI")?;
        if count>65535 || rows/128>65535 {return Err("tensor-core group grid exceeds the driver limit".into());}
        // Packed operands of the whole group: cols/2 bytes of codes and cols/16 bytes of scale codes a row.
        // Fully written by the pack kernel before the product kernel reads them.
        // They live in a kept scratch (grown when a group needs more): every earlier launch that read it has been
        // submitted on the one ordered stream, so a later pack launch overwrites it only after those launches ran.
        let need=[rows_in.checked_mul(cols/8).ok_or("tensor-core codes overflow")?,(rows_in*(cols/16)).div_ceil(4),(table.len()*std::mem::size_of::<NvPart>()).div_ceil(4),rows_in];
        let mut scratch=self.tc.scratch.borrow_mut();
        if scratch.as_ref().is_none_or(|kept|kept.iter().zip(need).any(|(v,n)|v.len()<n)) {
            // Kept operands must be there: a regrown scratch would not hold them.
            if matches!(operands,NvOperands::Kept) {return Err("tensor-core kept operands are absent".into());}
            // Round up so a run of growing groups reallocates a few times, not every call.
            let grown=need.map(|n|n.next_power_of_two().max(1<<16));
            *scratch=None;
            let mut fresh=[self.vector(grown[0])?,self.vector(grown[1])?,self.vector(grown[2])?,self.vector(grown[3])?];
            for v in fresh.iter_mut() {self.skip_fill(v)?;}
            *scratch=Some(fresh);
        }
        let [codes,scales,owner,row_scales]=scratch.as_mut().ok_or("tensor-core scratch is absent")?;
        let rows_abi=u32::try_from(rows_in).map_err(|_|"tensor-core rows exceed ABI")?;
        // The operand launch of this call: the pack of the quantizer's rows, the one-pass quantize-and-pack of FP32 rows, or none.
        let (mut pack,mut fused)=(None,None);
        match operands {
            NvOperands::Rows(input,stride)=>{
                if stride!=cols/16*17 || input.is_segmented() || input.len()<rows_in*stride || input.device_address()&3!=0 {return Err("tensor-core pack input extent differs".into());}
                pack=Some(NvPack {input:input.device_address(),codes:codes.device_address(),scales:scales.device_address(),rows:rows_abi,cols:cols as u32,stride:stride as u32,pad:0});
            },
            NvOperands::Fp32(input,activations)=>{
                if activations.len()!=rows_in || input.is_segmented() || input.len()<rows_in*cols || input.device_address()&3!=0 {return Err("tensor-core quantize-pack input extent differs".into());}
                self.write_vector_bytes(row_scales,&activations.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
                fused=Some(NvQuantizePack {input:input.device_address(),codes:codes.device_address(),scales:scales.device_address(),activations:row_scales.device_address(),rows:rows_abi,cols:cols as u32});
                self.require_arm_kernels(&["nv_quantize_pack"])?;crate::prompt_launches::record("nv_quantize_pack");
            },
            NvOperands::Kept=>{},
        }
        let pack_blocks=crate::tc_abi::nv_pack_blocks(rows_in,cols).ok_or("tensor-core pack grid overflow")?;
        let fused_blocks=crate::tc_abi::nv_quantize_pack_blocks(rows_in,cols).ok_or("tensor-core quantize-pack grid overflow")?;
        let mut bytes=Vec::with_capacity(table.len()*std::mem::size_of::<NvPart>());
        for e in table {
            for v in [e.weight,e.scales,output.device_address()+u64::from(e.first)*4] {bytes.extend_from_slice(&v.to_le_bytes());}
            for v in [e.first_row,e.shape.tokens] {bytes.extend_from_slice(&v.to_le_bytes());}
            for v in [e.shape.global,e.shape.activation] {bytes.extend_from_slice(&v.to_le_bytes());}
        }
        if bytes.len()!=table.len()*std::mem::size_of::<NvPart>() {return Err("tensor-core part table layout differs".into());}
        self.write_vector_bytes(owner,&bytes)?;
        let most=table.iter().map(|e|e.shape.tokens).max().unwrap_or(1);
        let mut args=NvRows {table:owner.device_address(),codes:codes.device_address(),scales:scales.device_address(),count,rows:rows as u32,cols:cols as u32,layout:1,skip:1,raw:0,chains};
        let (entry,block_tokens)=nv_rows_entry(chains);
        self.require_arm_kernels_tc(&[entry])?;if pack.is_some() {self.require_arm_kernels_tc(&["tc_nv_pack"])?;}
        crate::prompt_launches::record("fp4_cohort");
        // SAFETY: `nv_rows_tc_admitted` and the caller's span checks (`expert_group`) give every part column-tile
        // owners of rows * cols / 2 and rows * cols / 16 bytes, 16-byte aligned, an input span of tokens * stride
        // words inside the checked input, and an output span of rows * tokens words inside the group's private output;
        // parts own disjoint output spans. The output is NOT filled before the launch: the product kernel stores every
        // word of every part (one writing thread a word) and reads none of another thread's, and for second matrices
        // (`chains` 4) it parks partial sums in a thread's own words until its last step, so nothing may read the
        // output before the stream has run this launch; the caller hands it out only as a later consumer on the same
        // ordered stream. The packed operands hold rows_in rows in the kept scratch (`TcState::scratch`): it is reused by the
        // next call, which is sound because every launch is submitted on the one ordered stream, so a later pack
        // launch overwrites the scratch only after the earlier product launch has run; a regrown scratch drops the old
        // buffers, and a Buffer drop drains the stream first. The pack launch precedes the product launch on the
        // stream. Every owner outlives the stream drain of its Buffer drop. DIRECT launches, like the other group
        // kernels (the grid follows the group's token count).
        // THE OPERAND LAUNCH is one of three. `Rows`: `tc_nv_pack` on rows_in rows of the checked stride inside the checked
        // input (above). `Fp32`: `nv_quantize_pack` on rows_in rows of cols FP32 words inside the checked input, with
        // rows_in scale words written to the kept scratch just before (above); it stores every byte of the rows_in rows of
        // both operand arrays, as the pack does. `Kept`: no launch; the caller (`expert_group_quantized`) made an `Fp32`
        // call for the same rows and scales earlier in the same host call, nothing else touches the scratch between
        // (one thread, one ordered stream), and the scratch was not regrown (checked above).
        self.unrecorded(||unsafe {
            if let Some(pack)=pack.as_mut() {self.launch_flash_grid("tc_nv_pack",pack_blocks,1,1,NV_PACK_THREADS,0,&mut [parameter(pack)])?;}
            if let Some(fused)=fused.as_mut() {self.launch_flash_grid("nv_quantize_pack",fused_blocks,1,1,NV_QUANTIZE_PACK_THREADS,0,&mut [parameter(fused)])?;}
            self.launch_flash_grid(entry,most.div_ceil(block_tokens),(rows/128) as u32,count,NV_ROWS_THREADS,0,&mut [parameter(&mut args)])
        })?;
        Ok(())
    }
    fn require_arm_kernels_tc(&self,names:&[&str])->Result<(),String> {
        let module=self.tc_module()?;for name in names {module.function(name)?;}Ok(())
    }
}
