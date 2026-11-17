//! Block-FP8 decode projection (`Op::Fp8Serial`) on tensor cores: one token row a job. The word is the contract of
//! `tc_f8` (`docs/tc-dense-contract.md`, section Block FP8), made by the one body of `fp8_tc`: the same word as a row
//! group (`fp8_rows_small`, `fp8_rows_few`, `fp8_rows_multi`), the prompt tile (`fp8_rows_tile`), the one-warp body
//! (`serial_projection::project!`) and the host reference (`expert_ops::unit`).
//! A job with the role `FP8_QUARTER` is one column quarter of a second matrix: one range over the job's scale blocks,
//! stored with no round (`tc_twin_f8::range` is its CPU twin). The role `FP8_UNROUNDED` is the whole word with no round.
//!
//! Launch: the same `LaunchArguments` and job table as `expert_operation`; one block a tile of 16 rows of a job,
//! `fp8_tile_blocks(rows) * count` blocks (job-major); blocks of 4, 8, 16 or 32 whole warps (the host's choice,
//! `expert_ops::fp8_tile_block`: the word does not depend on it); dynamic shared `fp8_row_shared` bytes of the job
//! that needs most. The host launches this body only for `fp8_tile_admits(shape)` with 16-byte aligned weight and
//! input, and `expert_operation` selects it by the block size (`fp8_row_threads`; the one-warp body has 32 threads).
use crate::expert_ops::{Arguments,LaunchArguments};

/// The production form: the prefetch distance `FP8_ROW_PREFETCH` and stride `FP8_ROW_STRIDE`.
#[inline(always)]
pub(crate) fn tile(launch:LaunchArguments) {project::<{crate::expert_ops::FP8_ROW_PREFETCH},{crate::expert_ops::FP8_ROW_STRIDE},1>(launch)}
/// The decode tile as a kernel entry of its own: the launch record, grid and shared bytes of the block FP8 form of
/// `expert_operation` before it (`expert_ops::launch_geometry`). Recorded as `Kind::Fp8Tile` (`resident_launch_fp8_tile`).
#[no_mangle]
pub extern "ptx-kernel" fn fp8_tile(launch:LaunchArguments) {tile(launch)}

/// `PF`, `PS`: the prefetch distance and stride of the row body. `T` (1, 2 or 4; the served form is 1): 16-row tiles a block. Then a job has `ceil(rows / (16 T))` blocks, a block has `T` times the
/// warps and the shared bytes of one tile, and at least 4 warps a tile. The word does not depend on `T`.
#[inline(always)]
pub(crate) fn project<const PF:usize,const PS:usize,const T:u32>(launch:LaunchArguments) {
    // SAFETY: The host checks every job descriptor (shape, weight, scale and input owners and extents), keeps all
    // owners alive through stream completion, and fills the job table (count entries) for a batch. Weights, scales
    // and inputs are immutable; each (job, row) output word has exactly one writing thread (`fp8_tc::row_body`, a
    // store before any read of it) and the output is private; the host skips its fill (fill_free). The body traps
    // unless the block is 4, 8, 16 or 32 whole warps with at least 4 warps a tile, the dynamic shared region holds
    // the row body's bytes of `T` tiles for this job and one token row, the job count is 1 to FP8_TILE_MAX_JOBS, the descriptor is an admitted Fp8Serial shape
    // with the launch's row count, the weight and input owners are 16-byte aligned and the scales 4-byte aligned.
    // The dynamic region is the only shared allocation in the kernels that reach this body, so it starts at 0. A
    // block whose job or row tile does not exist returns before any barrier, all its threads alike.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x};
        let threads=_block_dim_x() as u32;let warps=threads/32/T;
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        let rows_launch=launch.single.shape.rows;
        if !crate::expert_ops::fp8_row_threads(threads) || !matches!(T,1|2|4) || warps<4 || !PS.is_power_of_two() || PS>4 || launch.count==0 || launch.count>crate::expert_ops::FP8_TILE_MAX_JOBS || rows_launch==0 {asm!("trap;");}
        // One job needs no division: its blocks are its row tiles.
        let per_job=rows_launch.div_ceil(crate::fp8_tc::ROWS as u32*T);let block=_block_idx_x() as u32;
        let (job,tile)=if launch.count==1 {(0,block)} else {(block/per_job,block%per_job)};
        if job>=launch.count || tile>=per_job {return;}
        let p:Arguments=if launch.batch==0 {launch.single} else {(launch.batch as *const Arguments).add(job as usize).read()};
        let shape=p.shape;
        if !crate::expert_ops::fp8_tile_admits(&shape) || shape.rows!=rows_launch || (p.weight|p.input[0])&15!=0 || p.scales&3!=0 {asm!("trap;");}
        let (raw,round)=crate::fp8_tc::flags(shape.role);
        if dynamic<T*crate::expert_ops::fp8_row_shared(warps,(shape.cols+shape.block_cols-1)>>shape.block_cols.trailing_zeros(),1,raw) {asm!("trap;");}
        let job=crate::fp8_tc::Job {weights:p.weight as *const u8,scales:p.scales as *const f32,rows:shape.rows as usize,cols:shape.cols as usize,
            block_rows:shape.block_rows as usize,block_cols:shape.block_cols as usize,row0:(tile*T) as usize*crate::fp8_tc::ROWS,input:p.input[0] as *const f32,stride:0,tokens:1,
            output:p.output as *mut f32,pitch:1,raw,round,tiles:T};
        crate::fp8_tc::row_body::<PF,PS>(&job,0);
    }
}
