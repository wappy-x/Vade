//! Owned complete MXFP8 rows. Calls use ordered queue retention or complete before borrowed operands can leave.
use super::*;
use crate::{mxfp8_projection::Shape,norm_boundary::WeightType};

pub struct ResidentMxfp8Projection<'b,'a> {backend:&'b Backend<'a>,shape:Shape,weights:Buffer<'a>,scales:Buffer<'a>}
pub struct Mxfp8ProjectionPass<'a> {accumulator:DeviceVector<'a>,output:Bf16Vector<'a>}
pub enum ResidentProjection<'v,'b,'a> {
    Mxfp8 {matrix:&'v ResidentMxfp8Projection<'b,'a>,input:&'v DeviceVector<'v>},
    Plain {rows:usize,cols:usize,weight:&'v DeviceVector<'v>,input:&'v DeviceVector<'v>},
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
            };
            self.check_vector(input,cols)?;if input.is_segmented() {return Err("resident projection batch input must be contiguous".into());}
            borrowed.push((input.device_address(),input.len()*4));
            let end=offset.checked_add(rows).ok_or("resident projection batch output overflow")?;
            if end>output.len() {return Err("resident projection batch output is short".into());}
            batch.push(MxProjection {weights:weight,scales,input:input.device_address(),output:output.device_address().checked_add((offset*4) as u64).ok_or("resident projection batch address overflow")?,
                rows:rows.try_into().map_err(|_|"resident projection batch rows exceed ABI")?,cols:cols.try_into().map_err(|_|"resident projection batch columns exceed ABI")?,tokens:1,input_bf16:0,round:0},plain)?;
            offset=end;
        }
        self.check_vector(output,offset)?;if output.is_segmented() {return Err("resident projection batch output must be contiguous".into());}
        let end=output.device_address().checked_add((offset*4) as u64).ok_or("resident projection batch output address overflow")?;
        for (start,bytes) in borrowed {let borrowed_end=start.checked_add(bytes as u64).ok_or("resident projection batch operand address overflow")?;
            if output.device_address()<borrowed_end && start<end {return Err("resident projection batch output aliases an operand".into());}
        }
        self.require_arm_kernels(&["flash_resident_projection_batch"])?;
        // No fill: the region is fully written by flash_resident_projection_batch before any read.
        self.skip_fill(output)?;self.resident_completion()?;
        // SAFETY: Every descriptor comes from checked immutable resident MXFP8
        // views or exact same-context contiguous FP32 owners. Prefix blocks keep
        // each original row schedule. Exact output length, checked address sums,
        // and retained private storage fund disjoint output spans. Each span is fully written by this kernel with raw stores before any read of it (fill_free). All borrowed
        // owners and the by-value descriptor batch live through stream completion.\n        // An owned queue defers dropped allocations through submission and its\n        // real drain. Ordinary callers retain the same completion guarantee.
        let launched=unsafe {self.launch_flash("flash_resident_projection_batch",batch.layer.ends[batch.layer.count as usize],128,&mut [parameter(&mut batch)])};
        let completed=self.resident_completion();launched.and(completed)
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
        // Which kernel runs is decided by small_choice::mx (its order is tested on the host): the small entries for 2 to 4 tokens of the listed shapes (the
        // tile with only that many tokens of work, lane F's mx_rows_small entries, same bytes), else lane K's tile when its predicate admits a piece, else
        // the old kernels. One token keeps the old M1 kernel (97 against 126 us at 4096 x 4096). Output rows are by token, so tile pieces of the largest
        // proven size simply follow each other.
        let piece=crate::mx_rows::MAX_TOKENS;
        let choice=crate::small_choice::mx(rows,cols,tokens,self.weights.address,input.device_address(),output.device_address());
        let small=if let crate::small_choice::Mx::Small(entry,per)=choice {Some((entry,per))} else {None};
        let tiled=choice==crate::small_choice::Mx::Tile;
        if let Some((entry,_))=small {b.require_arm_kernels(&[entry])?;} else if tiled {b.require_arm_kernels(&["mx_rows_tile"])?;}
        // SAFETY: Shape bounds complete E4M3 rows and UE8M0 groups in the retained arena views.
        // check_vector checks the context and all token rows, and the offsets stay inside them.
        // The output is a new disjoint allocation. Buffer::drop drains the stream, so all
        // owners outlive the launch. The argument is copied at launch. The small and the tile kernels take one
        // projection of all tokens (the small ones 2 to 4), and each traps unless the alignment its predicate checked holds.
        unsafe {
        if let Some((entry,_))=small {
            let [row_tiles,token_tiles]=crate::mx_small::grid(rows,tokens).ok_or("MXFP8 small grid overflow")?;
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
