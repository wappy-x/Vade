//! Owned complete MXFP8 rows. Calls use ordered queue retention or complete before borrowed operands can leave.
use super::*;
use crate::{mxfp8_projection::Shape,norm_boundary::WeightType};

pub struct ResidentMxfp8Projection<'b,'a> {backend:&'b Backend<'a>,shape:Shape,weights:Buffer<'a>,scales:Buffer<'a>}
pub struct Mxfp8ProjectionPass<'a> {accumulator:DeviceVector<'a>,output:Bf16Vector<'a>}
pub enum ResidentProjection<'v,'b,'a> {
    Mxfp8 {matrix:&'v ResidentMxfp8Projection<'b,'a>,input:&'v DeviceVector<'v>},
    Plain {rows:usize,cols:usize,weight:&'v DeviceVector<'v>,input:&'v DeviceVector<'v>},
    PlainBf16 {rows:usize,cols:usize,weight:&'v DeviceVector<'v>,input:&'v DeviceVector<'v>},
}
impl ResidentMxfp8Projection<'_,'_> {
    /// Output rows of this rank's matrix.
    pub fn rows(&self)->usize {self.shape.rows()}
    /// Bytes of the E4M3 rows, without scales.
    pub fn weight_bytes(&self)->usize {self.weights.bytes}
    /// CHAIN-SHADOW: the byte span `[offset, offset+bytes)` of the E4M3 rows.
    pub fn shadow_range(&self,offset:usize,bytes:usize)->Result<super::ShadowRange<'_>,String> {
        super::ShadowRange::checked(self.weights.address,self.weights.bytes,offset,bytes,self.backend.context)
    }
}
impl<'a> Backend<'a> {
    pub fn resident_projection_batch(&self,jobs:&[ResidentProjection<'_,'_,'a>],output:&mut DeviceVector<'_>,round:bool)->Result<(),String> {
        use crate::next_params::{MxProjection,ResidentProjectionBatch};
        self.require_resident_stream()?;
        if jobs.is_empty() || jobs.len()>9 {return Err("resident projection batch count differs".into());}
        let bf16=jobs.iter().all(|job|!matches!(job,ResidentProjection::Plain {..}));
        if !bf16 && jobs.iter().any(|job|matches!(job,ResidentProjection::PlainBf16 {..})) {return Err("resident projection batch mixes plain weight precisions".into());}
        let folded=bf16 && jobs.iter().all(|job|match job {
            ResidentProjection::PlainBf16 {rows,cols,weight,input}=>crate::plain_admission::fold_admitted(*rows,*cols,weight.device_address() as usize,input.device_address() as usize,crate::plain_admission::FOLD_THREADS),
            _=>false,
        });
        let mut batch=ResidentProjectionBatch {round:u32::from(round),..Default::default()};let mut offset=0usize;let mut borrowed=Vec::new();
        for job in jobs {
            let (rows,cols,weight,scales,input,plain)=match job {
                ResidentProjection::Mxfp8 {matrix,input}=>{
                    if !ptr::eq(matrix.backend.context,self.context) || matrix.shape.input()!=WeightType::F32 {return Err("resident MXFP8 batch context or input precision differs".into());}
                    borrowed.push((matrix.weights.address,matrix.weights.bytes));borrowed.push((matrix.scales.address,matrix.scales.bytes));
                    (matrix.shape.rows(),matrix.shape.cols(),matrix.weights.address,matrix.scales.address,*input,false)
                },
                ResidentProjection::Plain {rows,cols,weight,input}=>{
                    let words=rows.checked_mul(*cols).ok_or("resident plain batch weight overflow")?;self.check_vector(weight,words)?;
                    if weight.is_segmented() {return Err("resident plain batch weights must be contiguous".into());}
                    borrowed.push((weight.device_address(),weight.len()*4));
                    (*rows,*cols,weight.device_address(),4,*input,true)
                },
                ResidentProjection::PlainBf16 {rows,cols,weight,input}=>{
                    let elements=rows.checked_mul(*cols).ok_or("resident BF16 batch weight overflow")?;self.check_vector(weight,elements.div_ceil(2))?;
                    if weight.is_segmented() {return Err("resident BF16 batch weights must be contiguous".into());}
                    if !crate::plain_admission::deep_admitted(*rows,*cols,weight.device_address() as usize,input.device_address() as usize) {return Err("resident BF16 batch shape differs from the ordered reader envelope".into());}
                    borrowed.push((weight.device_address(),weight.len()*4));
                    (*rows,*cols,weight.device_address(),2,*input,true)
                },
            };
            self.check_vector(input,cols)?;if input.is_segmented() {return Err("resident projection batch input must be contiguous".into());}
            borrowed.push((input.device_address(),input.len()*4));
            let end=offset.checked_add(rows).ok_or("resident projection batch output overflow")?;
            if end>output.len() {return Err("resident projection batch output is short".into());}
            batch.push(MxProjection {weights:weight,scales,input:input.device_address(),output:output.device_address().checked_add((offset*4) as u64).ok_or("resident projection batch address overflow")?,
                rows:rows.try_into().map_err(|_|"resident projection batch rows exceed ABI")?,cols:cols.try_into().map_err(|_|"resident projection batch columns exceed ABI")?,tokens:1,input_bf16:0,round:0},plain)?;
            offset=end;
        }
        if folded {let mut end=0u32;for index in 0..batch.layer.count as usize {
            end=end.checked_add(batch.layer.projections[index].rows.div_ceil(crate::plain_admission::FOLD_ROWS as u32)).ok_or("resident BF16 fold grid overflow")?;batch.layer.ends[index+1]=end;
        }}
        self.check_vector(output,offset)?;if output.is_segmented() {return Err("resident projection batch output must be contiguous".into());}
        let end=output.device_address().checked_add((offset*4) as u64).ok_or("resident projection batch output address overflow")?;
        for (start,bytes) in borrowed {let borrowed_end=start.checked_add(bytes as u64).ok_or("resident projection batch operand address overflow")?;
            if output.device_address()<borrowed_end && start<end {return Err("resident projection batch output aliases an operand".into());}
        }
        let entry=if folded {"flash_resident_projection_plain_bf16"} else if bf16 {"flash_resident_projection_bf16"} else {"flash_resident_projection_batch"};
        self.require_arm_kernels(&[entry])?;
        // No fill: the region is fully written by flash_resident_projection_batch before any read.
        self.skip_fill(output)?;self.resident_completion()?;
        // SAFETY: Every descriptor comes from checked immutable resident MXFP8
        // views or exact same-context contiguous FP32/BF16 owners. BF16 rows
        // have half as many storage words; mixed precision is rejected. The
        // fold admits complete aligned rows, four rows per 160-thread block.
        // The other entries keep 128 threads and their original row schedules.
        // Exact output length, checked address sums,
        // and retained private storage fund disjoint output spans. Each span is fully written by this kernel with raw stores before any read of it (fill_free). All borrowed
        // owners and the by-value descriptor batch live through stream completion.
        // An owned queue defers dropped allocations through submission and its
        // real drain. Ordinary callers retain the same completion guarantee.
        let launched=unsafe {self.launch_flash(entry,batch.layer.ends[batch.layer.count as usize],if folded {crate::plain_admission::FOLD_THREADS as u32} else {128},&mut [parameter(&mut batch)])};
        let completed=self.resident_completion();launched.and(completed)
    }
    /// ROW STAGE: the merged launch of `resident_projection_batch` for `tokens` = 2 to 4 source rows (a verify pass). Each job
    /// reads `[tokens][cols]` FP32 words and writes `[tokens][rows]` words at its place in `output`, jobs in order. One launch
    /// reads every weight once for all source rows; every output word keeps the one-row entry's arithmetic and order
    /// (`flash_resident_projection_bf16_rows`; kernel-bench --rows-p1 compares every byte with the one-row entry).
    /// The caller admits by shape and storage alone (`plain_admission::rows_admitted`), so every rank makes the same choice;
    /// a job outside the entry's contract here is an error, never another path.
    pub fn resident_projection_batch_rows(&self,jobs:&[ResidentProjection<'_,'_,'a>],output:&mut DeviceVector<'_>,round:bool,tokens:usize)->Result<(),String> {
        use crate::next_params::{MxProjection,ResidentProjectionBatch};
        self.require_resident_stream()?;
        if jobs.is_empty() || jobs.len()>9 || !(2..=4).contains(&tokens) {return Err("resident projection row batch count differs".into());}
        let mut batch=ResidentProjectionBatch {round:u32::from(round),..Default::default()};let mut offset=0usize;let mut borrowed=Vec::new();
        for job in jobs {
            let (rows,cols,weight,scales,input,plain)=match job {
                ResidentProjection::Mxfp8 {matrix,input}=>{
                    if !ptr::eq(matrix.backend.context,self.context) || matrix.shape.input()!=WeightType::F32 {return Err("resident MXFP8 row batch context or input precision differs".into());}
                    borrowed.push((matrix.weights.address,matrix.weights.bytes));borrowed.push((matrix.scales.address,matrix.scales.bytes));
                    (matrix.shape.rows(),matrix.shape.cols(),matrix.weights.address,matrix.scales.address,*input,false)
                },
                ResidentProjection::Plain {..}=>return Err("resident projection row batch takes BF16 plain weights".into()),
                ResidentProjection::PlainBf16 {rows,cols,weight,input}=>{
                    let elements=rows.checked_mul(*cols).ok_or("resident BF16 row batch weight overflow")?;self.check_vector(weight,elements.div_ceil(2))?;
                    if weight.is_segmented() || !crate::plain_admission::deep_admitted(*rows,*cols,weight.device_address() as usize,input.device_address() as usize) {
                        return Err("resident BF16 row batch shape differs from the ordered reader envelope".into());
                    }
                    borrowed.push((weight.device_address(),weight.len()*4));
                    (*rows,*cols,weight.device_address(),2,*input,true)
                },
            };
            self.check_vector(input,tokens.checked_mul(cols).ok_or("resident projection row batch input overflow")?)?;
            if input.is_segmented() {return Err("resident projection row batch input must be contiguous".into());}
            borrowed.push((input.device_address(),input.len()*4));
            let end=rows.checked_mul(tokens).and_then(|words|offset.checked_add(words)).ok_or("resident projection row batch output overflow")?;
            if end>output.len() {return Err("resident projection row batch output is short".into());}
            let address=output.device_address().checked_add((offset*4) as u64).ok_or("resident projection row batch address overflow")?;
            batch.push_rows(MxProjection {weights:weight,scales,input:input.device_address(),output:address,
                rows:rows.try_into().map_err(|_|"resident projection row batch rows exceed ABI")?,
                cols:cols.try_into().map_err(|_|"resident projection row batch columns exceed ABI")?,tokens:tokens as u32,input_bf16:0,round:0},plain)?;
            offset=end;
        }
        self.check_vector(output,offset)?;if output.is_segmented() {return Err("resident projection row batch output must be contiguous".into());}
        let end=output.device_address().checked_add((offset*4) as u64).ok_or("resident projection row batch output address overflow")?;
        for (start,bytes) in borrowed {let borrowed_end=start.checked_add(bytes as u64).ok_or("resident projection row batch operand address overflow")?;
            if output.device_address()<borrowed_end && start<end {return Err("resident projection row batch output aliases an operand".into());}
        }
        let entry="flash_resident_projection_bf16_rows";self.require_arm_kernels(&[entry])?;
        // No fill: every word of the region is written by the entry before any read (a warp stores its weight row's `tokens`
        // words, a plain thread its one word; the grid covers every (weight row, source row) once).
        self.skip_fill(output)?;self.resident_completion()?;crate::prompt_launches::record("mixed_rows_stage");
        // SAFETY: Every descriptor comes from a checked immutable resident MXFP8 view or an exact same-context contiguous BF16
        // owner inside the ordered reader's envelope, with `tokens` in 2..=4, `tokens * cols` initialized input words and
        // `tokens * rows` output words, 16-byte aligned inputs and weights and cols a multiple of 32 (`push_rows` and
        // `MxLayer::push` refuse anything else). `push_rows` gives an MXFP8 descriptor rows.div_ceil(4) blocks and a plain
        // one tokens * rows.div_ceil(128) blocks: the entry's grid contract. Exact output length, checked address sums and
        // retained private storage fund disjoint output spans, each fully written before any read. All borrowed owners and
        // the by-value descriptor batch live through stream completion (the owned queue defers dropped allocations;
        // ordinary callers complete here).
        let launched=unsafe {self.launch_flash(entry,batch.layer.ends[batch.layer.count as usize],128,&mut [parameter(&mut batch)])};
        let completed=self.resident_completion();launched.and(completed)
    }
    /// The arena of `load_weight_arena` from bytes the caller already read. The parts cover the arena from 0 to `bytes`
    /// in order with no gap, so every byte is written (no fill is needed); they go through the upload staging in pieces of
    /// at most its first size, so the staging never grows. All copies are complete on return.
    pub fn upload_weight_arena(&self,bytes:usize,parts:&[(usize,&[u8])])->Result<WeightArena<'a>,String> {
        let mut end=0;
        for &(destination,part) in parts {
            if destination!=end {return Err("weight arena parts leave a gap or overlap".into());}
            end=destination.checked_add(part.len()).ok_or("weight arena part overflow")?;
        }
        if end!=bytes {return Err("weight arena parts do not cover the arena".into());}
        self.synchronize()?;
        let start=Instant::now();let buffer=Buffer::new(self.context,bytes)?;self.stats.borrow_mut().allocation_ns+=start.elapsed().as_nanos();
        let start=Instant::now();
        for &(destination,part) in parts {
            for (index,piece) in part.chunks(crate::staging::UPLOAD_BYTES).enumerate() {self.write_buffer(&buffer,destination+index*crate::staging::UPLOAD_BYTES,piece)?;}
        }
        self.synchronize()?;
        let mut stats=self.stats.borrow_mut();stats.weight_bytes+=bytes as u128;stats.weight_read_upload_ns+=start.elapsed().as_nanos();
        Ok(WeightArena {buffer:Rc::new(buffer)})
    }
    /// The arena holds E4M3 rows followed by their original UE8M0 scale rows.
    pub fn resident_mxfp8_projection(&self,arena:&WeightArena<'a>,shape:Shape)->Result<ResidentMxfp8Projection<'_,'a>,String> {
        self.require_resident_stream()?;
        if !ptr::eq(arena.buffer._context,self.context) || arena.bytes()!=shape.arena_bytes()? {
            return Err("MXFP8 arena context or complete row extent differs".into());
        }
        self.require_arm_kernels(&["flash_next_mxfp8_m1"])?;
        let weights=Buffer::view(&arena.buffer,0,shape.weight_bytes())?;
        let scales=Buffer::view(&arena.buffer,shape.weight_bytes(),shape.scale_bytes())?;
        Ok(ResidentMxfp8Projection {backend:self,shape,weights,scales})
    }
}
impl<'a> ResidentMxfp8Projection<'_,'a> {
    pub fn storage_bytes(&self)->[usize;2] {[self.weights.bytes,self.scales.bytes]}
    pub fn read_storage(&self,scale:bool,offset:usize,bytes:usize)->Result<Vec<u8>,String> {
        self.backend.read_storage_bytes(if scale {&self.scales} else {&self.weights},offset,bytes)
    }
    /// Preserve the completed norm's actual dtype. Do not round FP32 input lanes.
    /// Output rows stay local until the layer's declared gather is complete.
    pub fn apply(&self,input:&NormOutput<'_>)->Result<Mxfp8ProjectionPass<'a>,String> {
        let b=self.backend;let s=self.shape;b.require_resident_stream()?;
        let dtype=match input {NormOutput::Bf16(_)=>WeightType::Bf16,NormOutput::F32(_)=>WeightType::F32};
        if dtype!=s.input() {return Err("MXFP8 norm output dtype differs from the bound producer".into());}
        let input=input.vector();b.check_vector(input,s.cols())?;
        if input.is_segmented() {return Err("MXFP8 projection requires complete contiguous input features".into());}
        let accumulator=b.vector(s.rows())?;
        let mut args=crate::next_params::MxLayer::default();
        args.push(crate::next_params::MxProjection {weights:self.weights.address,scales:self.scales.address,
            input:input.device_address(),output:accumulator.device_address(),rows:s.rows() as u32,
            cols:s.cols() as u32,tokens:1,input_bf16:0,round:0})?;
        // SAFETY: Shape bounds complete E4M3 rows and UE8M0 groups in the retained arena views.
        // check_vector checks the input context and full feature extent. MxLayer checks alignment and ABI limits.
        // The accumulator is a new disjoint allocation. By-value arguments are copied at submission. Dropped owners are retained
        // by the active queue until its real drain; ordinary calls complete here.
        let launched=unsafe {b.launch_flash("flash_next_mxfp8_m1",args.ends[1],128,&mut [parameter(&mut args)])};
        let completed=b.resident_completion();launched.and(completed)?;
        let output=b.round_bf16_lanes(&accumulator)?;
        Ok(Mxfp8ProjectionPass {accumulator,output})
    }
}
impl<'a> Backend<'a> {
    /// The raw quarter accumulators of an MXFP8 second matrix (KDA out_proj, dense down) for one token row: one
    /// projection a column quarter that the owner holds, in quarter order (4, 2 or 1 pairs of matrix and input). Each
    /// matrix is the quarter as its own matrix (its columns of every row and their scale bytes); each input holds the
    /// quarter's columns. One launch of today's one-token kernel, which stores the raw FP32 sum of its 32 lanes; no
    /// round follows. The output is QUARTER-MAJOR, `[quarter][row]` FP32 words (CPU twin `tc_twin_quarter::mx_chain`).
    pub fn mxfp8_quarters(&self,jobs:&[(&ResidentMxfp8Projection<'_,'a>,&DeviceVector<'_>)])->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let first=jobs.first().ok_or("quarter projection batch is empty")?.0.shape;
        if crate::pair_split::quarters_entry(jobs.len()).is_none() || first.input()!=WeightType::F32 || !first.cols().is_multiple_of(256)
            || jobs.iter().any(|(matrix,_)|!ptr::eq(matrix.backend.context,self.context) || matrix.shape!=first) {
            return Err("MXFP8 quarter projection needs 4, 2 or 1 equal FP32-input quarter matrices of whole tiles".into());
        }
        let rows=first.rows();
        let output=self.vector(rows.checked_mul(jobs.len()).ok_or("MXFP8 quarter output overflow")?)?;
        let mut args=crate::next_params::MxLayer::default();
        for (index,(matrix,input)) in jobs.iter().enumerate() {
            self.check_vector(input,first.cols())?;
            if input.is_segmented() {return Err("MXFP8 quarter projection requires complete contiguous input features".into());}
            args.push(crate::next_params::MxProjection {weights:matrix.weights.address,scales:matrix.scales.address,input:input.device_address(),
                output:output.device_address()+(index*rows*4) as u64,rows:rows as u32,cols:first.cols() as u32,tokens:1,input_bf16:0,round:0})?;
        }
        // SAFETY: As `ResidentMxfp8Projection::apply`, for up to four projections. Each Shape bounds complete E4M3 rows and
        // UE8M0 groups in its retained arena views. check_vector checks each input's context and full feature extent. MxLayer
        // checks alignment and ABI limits. The output is a new allocation of `rows` words a projection: the spans are disjoint
        // and overlap no operand. By-value arguments are copied at submission. Dropped owners are retained by the active queue
        // until its real drain; ordinary calls complete here.
        let launched=unsafe {self.launch_flash("flash_next_mxfp8_m1",args.ends[jobs.len()],128,&mut [parameter(&mut args)])};
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
    }
}
impl<'a> ResidentMxfp8Projection<'_,'a> {
    /// Decoded FP32 token rows `[tokens,cols]`. Output accumulators are
    /// `[tokens,rows]`. Each (row,token) keeps the same element set, ascending
    /// FMA chain and shuffle tree as M=1. Groups use the existing M kernels.
    pub fn apply_rows(&self,input:&DeviceVector<'_>,tokens:usize)->Result<DeviceVector<'a>,String> {
        let b=self.backend;let s=self.shape;b.require_resident_stream()?;
        if s.input()!=WeightType::F32 {return Err("MXFP8 prompt rows require FP32 decoded activations".into());}
        let cols=s.cols();let rows=s.rows();b.check_vector(input,tokens.checked_mul(cols).ok_or("MXFP8 token rows overflow")?)?;
        if input.is_segmented() || tokens==0 {return Err("MXFP8 projection requires complete contiguous input features".into());}
        let output=b.vector(tokens*rows)?;let mut token=0;
        // Which kernel runs is decided by small_choice::mx (its order is tested on the host): the small entries for 2 to 8 tokens of the listed shapes (the
        // one-token kernel's load schedule for that many tokens, the mx_rows_small entries, same bytes), else lane K's tile when its predicate admits a piece, else
        // the old kernels. One token keeps the old M1 kernel (97 against 126 us at 4096 x 4096). Output rows are by token, so tile pieces of the largest
        // proven size simply follow each other.
        let piece=crate::mx_rows::token_limit(rows,cols);
        let choice=crate::small_choice::mx(rows,cols,tokens,self.weights.address,input.device_address(),output.device_address());
        let small=if let crate::small_choice::Mx::Small(entry,per)=choice {Some((entry,per))} else {None};
        let tiled=choice==crate::small_choice::Mx::Tile;
        if let Some((entry,_))=small {b.require_arm_kernels(&[entry])?;} else if tiled {b.require_arm_kernels(&["mx_rows_tile"])?;}
        // SAFETY: Shape bounds complete E4M3 rows and UE8M0 groups in the retained arena views.
        // check_vector checks the context and all token rows, and the offsets stay inside them.
        // The output is a new disjoint allocation. Buffer::drop drains the stream, so all
        // owners outlive the launch. The argument is copied at launch. The small and the tile kernels take one
        // projection of all tokens (the small ones 2 to 8), and each traps unless the alignment its predicate checked holds.
        unsafe {
        if let Some((entry,_))=small {
            let [row_tiles,token_tiles]=crate::mx_small::grid(rows,cols,tokens).ok_or("MXFP8 small grid overflow")?;
            let mut p=crate::next_params::MxProjection {weights:self.weights.address,scales:self.scales.address,input:input.device_address(),output:output.device_address(),rows:rows as u32,
                cols:cols as u32,tokens:tokens as u32,input_bf16:0,round:0};
            crate::prompt_launches::record("mx_rows_small");
            b.launch_flash_grid(entry,row_tiles,token_tiles,1,crate::mx_small::THREADS,0,&mut [parameter(&mut p)])?;
        } else if tiled {
            while token<tokens {let count=(tokens-token).min(piece);
                let [row_tiles,token_tiles]=crate::mx_rows::grid(rows,count).ok_or("MXFP8 tile grid overflow")?;
                let mut p=crate::next_params::MxProjection {weights:self.weights.address,scales:self.scales.address,input:input.device_address()+(token*cols*4) as u64,output:output.device_address()+(token*rows*4) as u64,rows:rows as u32,
                    cols:cols as u32,tokens:count as u32,input_bf16:0,round:0};
                crate::prompt_launches::record("mx_rows_tile");
                b.launch_flash_grid("mx_rows_tile",row_tiles,token_tiles,1,crate::mx_rows::THREADS,0,&mut [parameter(&mut p)])?;
                token+=count;
            }
        } else {
        while token<tokens {let count=(tokens-token).min(16);
            let mut args=crate::next_params::MxLayer::default();
            let p=crate::next_params::MxProjection {weights:self.weights.address,scales:self.scales.address,
                input:input.device_address()+(token*cols*4) as u64,output:output.device_address()+(token*rows*4) as u64,rows:rows as u32,
                cols:cols as u32,tokens:count as u32,input_bf16:0,round:0};
            let kernel=p.kernel();args.push(p)?;
            b.require_arm_kernels(&[kernel])?;
            crate::prompt_launches::record("mx_rows_groups");
            b.launch_flash(kernel,args.ends[1],128,&mut [parameter(&mut args)])?;
            token+=count;
        }
        }
        }
        Ok(output)
    }
}
impl<'a> Mxfp8ProjectionPass<'a> {
    pub fn accumulator(&self)->&DeviceVector<'a> {&self.accumulator}
    pub fn output(&self)->&Bf16Vector<'a> {&self.output}
    /// Keep the already-rounded output and release the completed accumulator.
    pub fn into_output(self)->Bf16Vector<'a> {self.output}
}
