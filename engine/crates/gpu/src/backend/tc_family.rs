//! The tensor-core kernel family: the second image (`sm_121a`, crate `spark-tc-kernels`), loaded when one of its
//! entries is first launched. The routed-expert NVFP4 products of a prompt group run here on contract (b)
//! (`tc_contract`): `tc_nv_pack` turns the quantizer's FP32 activation rows into packed operands, then `fp4_cohort`
//! runs the block-scaled tensor-core instruction, one launch a layer. A dense prompt group skips the quantizer's rows:
//! `nv_quantize_pack` (the production image) writes the same packed operands straight from the FP32 rows, once for all
//! the matrices that read them (`Backend::expert_group_quantized`).
use super::*;
use crate::expert_ops::{GroupEntry,Op};
use crate::tc_abi::{NvPack,NvPart,NvQuantizeGather,NvQuantizePack,NvRows,NV_PACK_THREADS,NV_QUANTIZE_PACK_THREADS,NV_ROWS_BLOCK_TOKENS,NV_ROWS_MAX_COLS,NV_ROWS_THREADS};

/// Where the packed operands of a product launch come from.
pub(super) enum NvOperands<'v,'d> {
    /// The quantizer's rows (17 FP32 words a block, the stride given): `tc_nv_pack` packs them.
    Rows(&'v DeviceVector<'d>,usize),
    /// FP32 activation rows and the input scale of each row: `nv_quantize_pack` quantizes and packs them in one pass.
    Fp32(&'v DeviceVector<'d>,&'v [f32]),
    /// GATHERED ROWS: FP32 activation rows that are still one row a token, the input row of each operand row (the table
    /// `Op::PermuteRows` would read) and the input scale of each operand row: `nv_quantize_gather` reads each row where it
    /// lies and writes the operands `Fp32` writes from the copied rows.
    Gathered(&'v DeviceVector<'d>,&'v DeviceVector<'d>,&'v [f32]),
    /// The kept scratch already holds this group's operands: an earlier list of the same call wrote them.
    Kept,
    /// PACKED MIDDLE: the set's packed operands in one vector, made before this call (`Backend::nv_quantize_pack_planes`
    /// of each rank's columns, set side by side): rows_in rows of cols / 8 code words, then rows_in rows of cols / 64
    /// scale words. They are copied to the set's rows of the kept scratch; no operand launch runs for the set.
    Packed(&'v DeviceVector<'d>),
}

/// WIDE COHORT: one row set of a product launch (`Backend::nv_rows_tc`): what one call of `nv_rows_tc_one` takes.
pub(super) struct NvSet<'t,'v,'d> {
    /// The set's admitted group table (`nv_rows_tc_admitted`), first rows counted inside the set.
    pub table:&'t [GroupEntry],
    /// Where the set's packed operands come from.
    pub operands:NvOperands<'v,'d>,
    /// The set's private output: part e's word (row, token) at unit `e.first + row * tokens + token`.
    pub output:&'v DeviceVector<'d>,
    /// The input rows the table covers (the end of its last part).
    pub rows_in:usize,
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
    /// Check tools only: true = a dense prompt group does not take `nv_quantize_gather` (the caller copies the rows and
    /// `nv_quantize_pack` reads the copy, the same operand bytes). false = the serving choice.
    gather_off:Cell<bool>,
    /// WIDE COHORT: true = the kept scratch was grown for a launch of several row sets (a wide prompt group, which runs
    /// only under a prompt lease). `nv_tc_scratch_return` drops that scratch when the lease is returned.
    wide:Cell<bool>,
    /// Check tools only: true = a native prompt cohort of several sets keeps one work interval a (expert, set) and
    /// copies an expert's weight tile once a set (the same words). false = the serving choice (PAIRED PARTS).
    paired_off:Cell<bool>,
}
impl TcState<'_> {
    pub(super) fn new()->Self {Self {module:std::cell::OnceCell::new(),off:Cell::new(false),scratch:RefCell::new(None),fused_off:Cell::new(false),gather_off:Cell::new(false),wide:Cell::new(false),paired_off:Cell::new(false)}}
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
    /// GATHERED ROWS: true when the ACTIVE primary image has the `nv_quantize_gather` entry (an older kernel file under a
    /// newer host lacks it: such a group copies its rows and keeps `nv_quantize_pack`).
    pub(super) fn nv_quantize_gather_present(&self)->bool {self.require_arm_kernels(&["nv_quantize_gather"]).is_ok()}
    /// Check tools only: false = a dense prompt group copies its rows into expert order and runs `nv_quantize_pack` on the
    /// copy (the same operand bytes), so a tool can time `nv_quantize_gather` against it and compare the words.
    pub fn set_nv_quantize_gather(&self,on:bool) {self.tc.gather_off.set(!on);}
    pub(super) fn nv_quantize_gather_off(&self)->bool {self.tc.gather_off.get()}
    /// Check tools only: false = a native prompt cohort of several sets runs one work interval a (expert, set), so a
    /// tool can time PAIRED PARTS against it and compare the words.
    pub fn set_prompt_paired(&self,on:bool) {self.tc.paired_off.set(!on);}
    pub(super) fn prompt_paired_off(&self)->bool {self.tc.paired_off.get()}
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
    /// PACKED MIDDLE: the packed operands `nv_quantize_pack` writes for FP32 rows of `cols` words, row r at the input scale
    /// `activations[r]` (one word a row, on the device), as two fresh vectors: rows * cols / 8 words of codes and
    /// rows * cols / 64 words of scale codes. `cols` is a multiple of 64, so both are whole words and the launch stores
    /// every byte of both (one thread a 16-column block; no fill before it). A block's bytes depend on its own 16 words and
    /// its row's scale only (`expert_ops::quantize_packed_with`), so the rows of ONE rank's columns of a matrix input pack
    /// to that rank's bytes of the whole rows' operands. `input`, `activations` and both results must stay alive until the
    /// stream has run the launch (a Buffer drop waits for it).
    pub fn nv_quantize_pack_planes(&self,input:&DeviceVector<'_>,cols:usize,activations:&DeviceVector<'_>)->Result<(DeviceVector<'a>,DeviceVector<'a>),String> {
        self.require_resident_stream()?;let rows=activations.len();
        if rows==0 || cols==0 || !cols.is_multiple_of(64) || input.is_segmented() || activations.is_segmented() || input.len()!=rows.checked_mul(cols).ok_or("quantize-pack input overflow")?
            || (input.device_address()|activations.device_address())&3!=0 {return Err("quantize-pack planes extent differs".into());}
        self.check_vector(input,input.len())?;self.check_vector(activations,rows)?;
        let (mut codes,mut scales)=(self.vector(rows*(cols/8))?,self.vector(rows*(cols/64))?);
        self.skip_fill(&mut codes)?;self.skip_fill(&mut scales)?;
        let mut args=NvQuantizePack {input:input.device_address(),codes:codes.device_address(),scales:scales.device_address(),activations:activations.device_address(),
            rows:u32::try_from(rows).map_err(|_|"quantize-pack rows exceed ABI")?,cols:u32::try_from(cols).map_err(|_|"quantize-pack columns exceed ABI")?};
        let blocks=crate::tc_abi::nv_quantize_pack_blocks(rows,cols).ok_or("quantize-pack grid overflow")?;
        self.require_arm_kernels(&["nv_quantize_pack"])?;crate::prompt_launches::record("nv_quantize_pack");crate::prompt_launches::record("packed_middle");
        // The checks above give the kernel its contract: rows * cols input words, rows scale words, private outputs of
        // rows * cols / 2 and rows * cols / 16 bytes (whole words), 4-byte aligned; the caller keeps all four alive.
        self.unrecorded(||self.launch_checked_expert_payload("nv_quantize_pack",blocks,1,NV_QUANTIZE_PACK_THREADS,0,&mut args))?;
        Ok((codes,scales))
    }
    /// WIDE COHORT: a prompt lease is returned. Drops the kept scratch when a launch of several row sets grew it (the
    /// operands of every run of a wide group: about 19 MB a run of 1,024 rows on the model), so bytes that a wide
    /// group borrowed do not stay with the service; the next group allocates a scratch of its own size, as the first
    /// group after startup does. The drop of a buffer drains the stream first, so no launch still reads the scratch.
    pub fn nv_tc_scratch_return(&self) {if self.tc.wide.replace(false) {*self.tc.scratch.borrow_mut()=None;}}
    /// The routed-expert products of one admitted group table into `output`: `nv_rows_tc` for one row set.
    pub(super) fn nv_rows_tc_one(&self,table:&[GroupEntry],operands:NvOperands<'_,'_>,output:&DeviceVector<'_>,rows_in:usize,chains:u32)->Result<(),String> {
        self.nv_rows_tc(&[NvSet {table,operands,output,rows_in}],chains)
    }
    /// The routed-expert products of one or more row sets in ONE product launch. A set is one admitted group table
    /// into its `output` (the group's private output; this path writes every word of it before any read, so the
    /// caller does not fill it): part e's word (row, token) lands at unit `e.first + row * tokens + token`, as every
    /// group kernel writes it. `rows_in` = the input rows the table covers (the end of the last part). `operands`
    /// says how the packed operands of the `rows_in` input rows reach the kept scratch before the product launch.
    /// WIDE COHORT, two or more sets: each set has its own admitted table, its own operands and its own private output. The sets' packed
    /// operands lie one after the other in the kept scratch (set s starts at the row count of the sets before it),
    /// each written by its own operand launch, and the one product launch runs every part of every set. A part's
    /// record holds its own weights, its own first operand row, its own token count and its own output address, and
    /// the product kernel reads nothing else of a part (`fp4_cohort`: a word does not depend on its tile, on the
    /// cohort size or on the token's place), so every word of every set is the word its own launch stores.
    /// The parts of the sets that read the SAME expert follow each other in the part table: their blocks run side by
    /// side and read that expert's weights from memory once, where a launch a set reads them once a set. That is the
    /// whole gain: a prompt group of 1,024 rows touches every routed expert of a layer, so its product launches are
    /// bound by the weight read, not by the products. One set: the table order, the operand launch and the product
    /// launch are what they always were.
    pub(super) fn nv_rows_tc(&self,sets:&[NvSet<'_,'_,'_>],chains:u32)->Result<(),String> {
        let head=sets.first().and_then(|set|set.table.first()).ok_or("tensor-core group is empty")?;
        let (rows,cols)=(head.shape.rows as usize,head.shape.cols as usize);
        if sets.iter().any(|set|set.table.first().is_none_or(|e|e.shape.rows as usize!=rows || e.shape.cols as usize!=cols)) {return Err("tensor-core row sets differ in shape".into());}
        let parts=sets.iter().map(|set|set.table.len()).sum::<usize>();
        let count=u32::try_from(parts).map_err(|_|"tensor-core group exceeds ABI")?;
        if count>65535 || rows/128>65535 {return Err("tensor-core group grid exceeds the driver limit".into());}
        // The first operand row of each set, and the rows of all sets.
        let mut starts=Vec::with_capacity(sets.len());let mut rows_all=0usize;
        for set in sets {starts.push(rows_all);rows_all=rows_all.checked_add(set.rows_in).ok_or("tensor-core rows overflow")?;}
        let rows_abi=|rows:usize|u32::try_from(rows).map_err(|_|"tensor-core rows exceed ABI".to_string());
        rows_abi(rows_all)?;
        // Kept operands serve every set of a call or none: the sets of an earlier list of the same call wrote them all.
        let kept=sets.iter().filter(|set|matches!(set.operands,NvOperands::Kept)).count();
        if kept!=0 && kept!=sets.len() {return Err("tensor-core row sets mix kept and fresh operands".into());}
        // Packed operands of the whole group: cols/2 bytes of codes and cols/16 bytes of scale codes a row.
        // Fully written by the pack kernel before the product kernel reads them.
        // They live in a kept scratch (grown when a group needs more): every earlier launch that read it has been
        // submitted on the one ordered stream, so a later pack launch overwrites it only after those launches ran.
        let need=[rows_all.checked_mul(cols/8).ok_or("tensor-core codes overflow")?,(rows_all*(cols/16)).div_ceil(4),(parts*std::mem::size_of::<NvPart>()).div_ceil(4),rows_all];
        let mut scratch=self.tc.scratch.borrow_mut();
        if scratch.as_ref().is_none_or(|kept|kept.iter().zip(need).any(|(v,n)|v.len()<n)) {
            // Kept operands must be there: a regrown scratch would not hold them.
            if kept!=0 {return Err("tensor-core kept operands are absent".into());}
            // Round up so a run of growing groups reallocates a few times, not every call.
            let grown=need.map(|n|n.next_power_of_two().max(1<<16));
            *scratch=None;
            let mut fresh=[self.vector(grown[0])?,self.vector(grown[1])?,self.vector(grown[2])?,self.vector(grown[3])?];
            for v in fresh.iter_mut() {self.skip_fill(v)?;}
            *scratch=Some(fresh);
            // WIDE COHORT: a scratch grown for several sets is borrowed with the group's lease (`nv_tc_scratch_return`).
            if sets.len()>1 {self.tc.wide.set(true);}
        }
        let [codes,scales,owner,row_scales]=scratch.as_mut().ok_or("tensor-core scratch is absent")?;
        // The operand launch of each set: the pack of the quantizer's rows, the one-pass quantize-and-pack of FP32 rows
        // (copied or read through a table), or none. A set's operands start at its first row in both operand arrays
        // (cols/2 and cols/16 bytes a row: whole words, cols is a multiple of 64) and in the row scales.
        let (mut packs,mut fused,mut gathered)=(Vec::new(),Vec::new(),Vec::new());
        let mut row_scale_words=vec![0f32;if kept==0 {rows_all} else {0}];let mut scaled=false;
        for (set,&start) in sets.iter().zip(&starts) {
            let rows_in=set.rows_in;
            let (code_at,scale_at,activation_at)=(codes.device_address()+(start*(cols/2)) as u64,scales.device_address()+(start*(cols/16)) as u64,row_scales.device_address()+(start*4) as u64);
            match &set.operands {
                NvOperands::Rows(input,stride)=>{
                    if *stride!=cols/16*17 || input.is_segmented() || input.len()<rows_in*stride || input.device_address()&3!=0 {return Err("tensor-core pack input extent differs".into());}
                    packs.push((NvPack {input:input.device_address(),codes:code_at,scales:scale_at,rows:rows_abi(rows_in)?,cols:cols as u32,stride:*stride as u32,pad:0},crate::tc_abi::nv_pack_blocks(rows_in,cols).ok_or("tensor-core pack grid overflow")?));
                },
                NvOperands::Fp32(input,activations)=>{
                    if activations.len()!=rows_in || input.is_segmented() || input.len()<rows_in*cols || input.device_address()&3!=0 {return Err("tensor-core quantize-pack input extent differs".into());}
                    row_scale_words[start..start+rows_in].copy_from_slice(activations);scaled=true;
                    fused.push((NvQuantizePack {input:input.device_address(),codes:code_at,scales:scale_at,activations:activation_at,rows:rows_abi(rows_in)?,cols:cols as u32},crate::tc_abi::nv_quantize_pack_blocks(rows_in,cols).ok_or("tensor-core quantize-pack grid overflow")?));
                    self.require_arm_kernels(&["nv_quantize_pack"])?;crate::prompt_launches::record("nv_quantize_pack");
                },
                NvOperands::Gathered(input,index,activations)=>{
                    // Whole input rows of `cols` words, and one index word and one scale word for each of the rows_in operand rows.
                    if activations.len()!=rows_in || index.len()<rows_in || index.is_segmented() || input.is_segmented() || input.is_empty() || !input.len().is_multiple_of(cols)
                        || (input.device_address()|index.device_address())&3!=0 {return Err("tensor-core quantize-gather input extent differs".into());}
                    row_scale_words[start..start+rows_in].copy_from_slice(activations);scaled=true;
                    gathered.push((NvQuantizeGather {input:input.device_address(),codes:code_at,scales:scale_at,activations:activation_at,index:index.device_address(),
                        rows:rows_abi(rows_in)?,cols:cols as u32,source_rows:u32::try_from(input.len()/cols).map_err(|_|"tensor-core source rows exceed ABI")?,pad:0},crate::tc_abi::nv_quantize_pack_blocks(rows_in,cols).ok_or("tensor-core quantize-pack grid overflow")?));
                    self.require_arm_kernels(&["nv_quantize_gather"])?;crate::prompt_launches::record("nv_quantize_gather");
                },
                NvOperands::Kept=>{},
                NvOperands::Packed(packed)=>{
                    // Whole words a row in both arrays (cols a multiple of 64), and exactly the set's rows.
                    let (code_words,scale_words)=(rows_in*(cols/8),rows_in*(cols/64));
                    if !cols.is_multiple_of(64) || packed.is_segmented() || packed.len()!=code_words+scale_words {return Err("tensor-core packed operand extent differs".into());}
                    // Two device copies on the one ordered stream, before the product launch: the set's rows of both
                    // operand arrays, every byte of them (safe calls: `copy_range` checks both spans).
                    self.unrecorded(||{self.copy_range(packed,codes,0,start*(cols/8),code_words)?;self.copy_range(packed,scales,code_words,start*(cols/64),scale_words)})?;
                },
            }
        }
        // The input scale of every row of every set, in one write before the operand launches that read them.
        if scaled {self.write_vector_bytes(row_scales,&row_scale_words.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;}
        // The parts in launch order: by the first appearance of their weights (set 0's parts in table order, then any
        // expert only a later set reads), and for one expert in set order. One set: its table order.
        let mut rank=std::collections::BTreeMap::new();
        for set in sets {for e in set.table {let next=rank.len();rank.entry(e.weight).or_insert(next);}}
        let mut order=Vec::with_capacity(parts);
        for (at,set) in sets.iter().enumerate() {for (index,e) in set.table.iter().enumerate() {order.push((rank[&e.weight],at,index));}}
        order.sort_unstable();
        let mut bytes=Vec::with_capacity(parts*std::mem::size_of::<NvPart>());let mut most=1;
        for &(_,at,index) in &order {
            let (set,e)=(&sets[at],&sets[at].table[index]);
            let first_row=rows_abi(starts[at].checked_add(e.first_row as usize).ok_or("tensor-core rows overflow")?)?;
            for v in [e.weight,e.scales,set.output.device_address()+u64::from(e.first)*4] {bytes.extend_from_slice(&v.to_le_bytes());}
            for v in [first_row,e.shape.tokens] {bytes.extend_from_slice(&v.to_le_bytes());}
            for v in [e.shape.global,e.shape.activation] {bytes.extend_from_slice(&v.to_le_bytes());}
            most=most.max(e.shape.tokens);
        }
        if bytes.len()!=parts*std::mem::size_of::<NvPart>() {return Err("tensor-core part table layout differs".into());}
        self.write_vector_bytes(owner,&bytes)?;
        let mut args=NvRows {table:owner.device_address(),codes:codes.device_address(),scales:scales.device_address(),count,rows:rows as u32,cols:cols as u32,layout:1,skip:1,raw:0,chains,work:0,work_count:0,merge:0,rowmap:0};
        let (entry,block_tokens)=nv_rows_entry(chains);
        self.require_arm_kernels_tc(&[entry])?;if !packs.is_empty() {self.require_arm_kernels_tc(&["tc_nv_pack"])?;}
        crate::prompt_launches::record("fp4_cohort");
        if sets.len()>1 {crate::prompt_launches::record("wide_cohort");}
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
        // `Gathered`: `nv_quantize_gather` on rows_in operand rows; the input holds whole rows of cols FP32 words (checked
        // above) and the kernel traps on an index word at or beyond that row count before it reads the input; rows_in
        // index words inside the checked index vector and rows_in scale words in the kept scratch; it stores every byte of
        // the rows_in rows of both operand arrays, as `Fp32` does. The index vector is the caller's upload: it is alive at
        // the launch, and its Buffer drop waits for the stream like every other owner here.
        // THE OPERAND LAUNCH is one of four. `Rows`: `tc_nv_pack` on rows_in rows of the checked stride inside the checked
        // input (above). `Fp32`: `nv_quantize_pack` on rows_in rows of cols FP32 words inside the checked input, with
        // rows_in scale words written to the kept scratch just before (above); it stores every byte of the rows_in rows of
        // both operand arrays, as the pack does. `Kept`: no launch; the caller (`expert_group_quantized`) made an `Fp32`
        // call for the same rows and scales earlier in the same host call, nothing else touches the scratch between
        // (one thread, one ordered stream), and the scratch was not regrown (checked above).
        // SEVERAL SETS (WIDE COHORT): each set's table passed `nv_rows_tc_admitted` and its caller's span checks on its
        // own, so the lines above hold for each set. Set s owns rows `starts[s] .. starts[s] + rows_in` of the two
        // operand arrays and of the row scales: the spans are disjoint and inside the scratch (`need` is the sum over
        // the sets, checked above), each operand launch stores only its own set's rows, and a part's first row is its
        // set's start plus its own first row, so a part reads only its own set's operands. Each part stores into its own
        // set's output, at its own span there: the outputs are different private vectors, so the parts of all sets own
        // disjoint output spans. Every operand launch precedes the one product launch on the stream.
        self.unrecorded(||unsafe {
            for (pack,blocks) in packs.iter_mut() {self.launch_flash_grid("tc_nv_pack",*blocks,1,1,NV_PACK_THREADS,0,&mut [parameter(pack)])?;}
            for (fused,blocks) in fused.iter_mut() {self.launch_flash_grid("nv_quantize_pack",*blocks,1,1,NV_QUANTIZE_PACK_THREADS,0,&mut [parameter(fused)])?;}
            for (gathered,blocks) in gathered.iter_mut() {self.launch_flash_grid("nv_quantize_gather",*blocks,1,1,NV_QUANTIZE_PACK_THREADS,0,&mut [parameter(gathered)])?;}
            self.launch_flash_grid(entry,most.div_ceil(block_tokens),(rows/128) as u32,count,NV_ROWS_THREADS,0,&mut [parameter(&mut args)])
        })?;
        Ok(())
    }
    pub(super) fn require_arm_kernels_tc(&self,names:&[&str])->Result<(),String> {
        let module=self.tc_module()?;for name in names {module.function(name)?;}Ok(())
    }
}
