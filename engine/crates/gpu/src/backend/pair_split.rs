//! The pair split's projection launches: the raw quarter accumulators of a second matrix that is stored by input
//! columns, and the expert fold inside each quarter. Shapes: crate::pair_split. CPU twin: crate::tc_twin_fold.
use super::*;
use crate::expert_ops::{LaunchArguments,Op,Shape,FP8_QUARTER,NV_SECOND};
use crate::pair_split::{QuarterFold,FOLD_MAX_EXPERTS,FOLD_PLAIN,FOLD_QUARTER_MAJOR,FOLD_ROUND,FOLD_SCALES_BY_ID,THREADS};
impl<'a> Backend<'a> {
    /// The raw quarter accumulators of equal one-token decode jobs (shape, input, weight, scales) of second matrices
    /// in column tiles. Each job's stored matrix holds `quarters` (4, 2 or 1) whole column quarters. The output is
    /// `[job][row][quarter]` FP32 words: a chain a quarter from +0.0, with no join, no epilogue and no round.
    pub fn nv_quarters(&self,jobs:&[(Shape,&DeviceVector<'_>,&DeviceVector<'_>,&DeviceVector<'_>)],quarters:usize)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let first=jobs.first().ok_or("quarter projection batch is empty")?.0;
        let entry=crate::pair_split::quarters_entry(quarters).ok_or("an owner holds 4, 2 or 1 column quarters")?;
        if first.op!=Op::NvProject || first.role!=NV_SECOND || first.tokens!=1 || !crate::expert_ops::batch_shapes_agree(jobs.iter().map(|job|job.0))
            || jobs.iter().any(|job|job.0.tokens!=1) || !(first.cols as usize/64).is_multiple_of(quarters) {
            return Err("quarter projection needs equal one-token second matrices of whole quarters".into());
        }
        let count=u32::try_from(jobs.len()).map_err(|_|"quarter projection batch exceeds ABI")?;
        let mut descriptors=jobs.iter().map(|(shape,input,weight,scales)|self.expert_arguments(*shape,[Some(input),None,None,None],Some(weight),Some(scales))).collect::<Result<Vec<_>,_>>()?;
        let owners=descriptors.iter().map(|job|(job.weight,job.scales)).collect::<Vec<_>>();
        if !crate::expert_ops::nv_cols_admitted(&first,count,&owners) || descriptors.iter().any(|job|job.input[0]&3!=0) {
            return Err("quarter projection is outside the column-tile decode launch".into());
        }
        let each=(first.rows as usize).checked_mul(quarters).ok_or("quarter projection output overflow")?;
        let words=each.checked_mul(jobs.len()).filter(|&words|words<=u32::MAX as usize && words<=isize::MAX as usize/4).ok_or("quarter projection output overflow")?;
        // No fill: every row of every job writes its `quarters` words before any read (rows are whole tiles).
        let mut output=self.vector(words)?;self.skip_fill(&mut output)?;
        for (index,arguments) in descriptors.iter_mut().enumerate() {arguments.output=output.device_address()+(index*each*4) as u64;}
        let table=self.upload_vector(&descriptors.iter().flat_map(|arguments|arguments.descriptor_words()).map(f32::from_bits).collect::<Vec<_>>())?;
        let blocks=crate::expert_ops::nv_cols_blocks(first.rows,count).ok_or("quarter projection grid overflow")?;
        self.require_arm_kernels(&[entry])?;
        let mut arguments=LaunchArguments {single:descriptors[0],batch:table.device_address(),count};
        let launched=self.launch_checked_expert_payload(entry,blocks,1,crate::expert_ops::NV_COLS_THREADS,crate::expert_ops::nv_cols_shared(first.cols),&mut arguments);
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
    }
    /// The raw quarter accumulators of a block-FP8 second matrix (shared expert down, latent o) for one token row:
    /// one decode job (shape, input, weight, scales) a column quarter that the owner holds, in quarter order (4, 2 or
    /// 1 jobs). Each job is the quarter as its own matrix with the role `FP8_QUARTER`, its own columns of the input
    /// and its own scale blocks. One launch of the tile kernel. The output is QUARTER-MAJOR, `[quarter][row]` FP32
    /// words: today's chain over the quarter from +0.0, with no round (CPU twin `tc_twin_quarter::fp8_chain`).
    pub fn fp8_quarters(&self,jobs:&[(Shape,&DeviceVector<'_>,&DeviceVector<'_>,&DeviceVector<'_>)])->Result<DeviceVector<'a>,String> {
        let first=jobs.first().ok_or("quarter projection batch is empty")?.0;
        if crate::pair_split::quarters_entry(jobs.len()).is_none() || first.op!=Op::Fp8Serial || first.role!=FP8_QUARTER || first.tokens!=1
            || !crate::expert_ops::batch_shapes_agree(jobs.iter().map(|job|job.0)) {
            return Err("FP8 quarter projection needs 4, 2 or 1 equal one-token raw quarter jobs".into());
        }
        // launch_geometry refuses these jobs unless the tile kernel takes them: no other body has the raw form.
        self.expert_projection_batch(jobs)
    }
    /// The expert fold of one token row inside each of the owner's `quarters` column quarters, and the join of those
    /// quarters (`nv_quarter_fold`). `ids`, `weights`, `scales`: one FP32 word a routed expert, in slot order.
    /// With `by_id`, `scales` holds one word for every expert of the layer in id order, and the kernel picks by the
    /// device's ids (the engine's form: the host does not read the route).
    /// `acc`: `[slot][row][quarter]` raw quarter accumulators (`nv_quarters`). `shared`: the shared expert's raw
    /// quarter accumulators, `[row][quarter]`, or `[quarter][row]` with `shared_major` (`fp8_quarters`). The output is
    /// `rows` words: the owner's partial sum, or with `round` (one owner) the finished BF16 word.
    #[allow(clippy::too_many_arguments)]
    pub fn nv_quarter_fold(&self,ids:&DeviceVector<'_>,weights:&DeviceVector<'_>,scales:&DeviceVector<'_>,by_id:bool,acc:&DeviceVector<'_>,shared:Option<&DeviceVector<'_>>,shared_major:bool,
        rows:usize,quarters:usize,round:bool)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let experts=ids.len();
        if experts==0 || experts>FOLD_MAX_EXPERTS as usize || rows==0 || rows>u32::MAX as usize || crate::pair_split::quarters_entry(quarters).is_none() {return Err("quarter fold shape differs".into());}
        let each=rows.checked_mul(quarters).ok_or("quarter fold extent overflow")?;
        let pool=if by_id {scales.len()} else {experts};
        if weights.len()!=experts || scales.len()!=pool || pool==0 || pool>crate::device_route::MAX_EXPERTS as usize
            || experts.checked_mul(each)!=Some(acc.len()) || shared.is_some_and(|shared|shared.len()!=each) {return Err("quarter fold extent differs".into());}
        for vector in [ids,weights,scales,acc].into_iter().chain(shared) {
            self.check_vector(vector,vector.len())?;if vector.is_segmented() {return Err("quarter fold operands must be contiguous".into());}
        }
        // No fill: every output row is written before any read. A fresh allocation overlaps no input.
        let mut output=self.vector(rows)?;self.skip_fill(&mut output)?;
        self.require_arm_kernels(&["nv_quarter_fold"])?;
        let mut arguments=QuarterFold {ids:ids.device_address(),weights:weights.device_address(),scales:scales.device_address(),acc:acc.device_address(),
            shared:shared.map_or(0,|shared|shared.device_address()),output:output.device_address(),experts:experts as u32,rows:rows as u32,quarters:quarters as u32,
            flags:(if round {FOLD_ROUND} else {0})|(if shared_major {FOLD_QUARTER_MAJOR} else {0})|(if by_id {FOLD_SCALES_BY_ID} else {0}),pool:pool as u32};
        let launched=self.launch_checked_expert_payload("nv_quarter_fold",crate::pair_split::blocks(rows) as u32,1,THREADS,0,&mut arguments);
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
    }
}
impl<'a> Backend<'a> {
    /// A second matrix with no fold: the join of each row's `quarters` raw quarter accumulators. `acc` is
    /// `[row][quarter]` (one job of `nv_quarters`), or `[quarter][row]` with `major` (`fp8_quarters`, or one MXFP8
    /// projection a quarter). The output is `rows` words: the owner's partial sum.
    pub fn nv_quarter_join(&self,acc:&DeviceVector<'_>,major:bool,rows:usize,quarters:usize)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        if rows==0 || rows>u32::MAX as usize || crate::pair_split::quarters_entry(quarters).is_none() || rows.checked_mul(quarters)!=Some(acc.len()) {return Err("quarter join extent differs".into());}
        self.check_vector(acc,acc.len())?;if acc.is_segmented() {return Err("quarter join operands must be contiguous".into());}
        // No fill: every output row is written before any read. A fresh allocation overlaps no input.
        let mut output=self.vector(rows)?;self.skip_fill(&mut output)?;
        self.require_arm_kernels(&["nv_quarter_fold"])?;
        let address=acc.device_address();
        let mut arguments=QuarterFold {ids:address,weights:address,scales:address,acc:address,shared:0,output:output.device_address(),experts:1,rows:rows as u32,quarters:quarters as u32,
            flags:FOLD_PLAIN|(if major {FOLD_QUARTER_MAJOR} else {0}),pool:0};
        let launched=self.launch_checked_expert_payload("nv_quarter_fold",crate::pair_split::blocks(rows) as u32,1,THREADS,0,&mut arguments);
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
    }
}
