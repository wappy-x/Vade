//! Admission and launch geometry for `nv_rows_stream`, the coalesced NVFP4 batch projection.
//! Shape and alignment only. The kernel is byte-equal to the packed `expert_operation` NvProject batch on the
//! family below (device sweep, benchmarks/2026-11-05/lane-k/nv-final-*); any other shape keeps `expert_operation`.
pub const ROWS_PER_BLOCK: usize = 64;
pub const MAX_ROWS: usize = 2048;
pub const MAX_COLS: usize = 8192;
pub const MAX_JOBS: usize = 16;
/// The decode tile body (`nv_tile`): a block owns TILE_ROWS rows of one job; 32 chain/idle threads plus
/// TILE_ROWS*8 producer threads; dynamic shared TILE_SHARED bytes = 4 stages of (16 rows x 64 weight bytes + 16 x 8 scale
/// bytes + 8 activation blocks of 80 bytes) + 2 product buffers of 16 rows x 132 words. Admission is `admitted`
/// (cols is a multiple of 512, so of the tile's 128-column chunk).
pub const TILE_ROWS: usize = 16;
pub const TILE_THREADS: u32 = 32 + TILE_ROWS as u32 * 8;
pub const TILE_SHARED: u32 = (4 * (TILE_ROWS * 64 + TILE_ROWS * 8 + 8 * 80) + 2 * TILE_ROWS * 132 * 4) as u32;
/// Blocks of a tile launch: job-major, `ceil(rows/16)` per job.
pub fn tile_blocks(rows: usize, jobs: usize) -> Option<u32> { u32::try_from(rows.div_ceil(TILE_ROWS).checked_mul(jobs)?).ok() }
/// True when a batch of `jobs` equal NvProject jobs of `rows` x `cols` may use `nv_rows_stream`.
/// `weight` and `scales` are the device addresses of every job's raw owners.
pub fn admitted(rows: usize, cols: usize, jobs: usize, owners: &[(u64, u64)]) -> bool {
    (1..=MAX_ROWS).contains(&rows) && (512..=MAX_COLS).contains(&cols) && cols.is_multiple_of(512) && (1..=MAX_JOBS).contains(&jobs)
        && owners.len() == jobs && owners.iter().all(|(w, s)| (w | s) & 15 == 0)
}
/// Blocks of 64 threads: job-major, `ceil(rows/64)` per job.
pub fn blocks(rows: usize, jobs: usize) -> Option<u32> { u32::try_from(rows.div_ceil(ROWS_PER_BLOCK).checked_mul(jobs)?).ok() }
/// Production also requires the original-arithmetic proof for the packed reference.
pub fn projection_admitted(shape:crate::expert_ops::Shape,jobs:u32,owners:&[(u64,u64)])->bool {
    shape.op==crate::expert_ops::Op::NvProject && !crate::expert_ops::nv_tiled(&shape) && crate::expert_ops::nv_packed_shape(shape.rows,shape.cols,jobs)
        && admitted(shape.rows as usize,shape.cols as usize,jobs as usize,owners)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_batches_are_admitted() {
        assert!(admitted(1024, 4096, 16, &[(256, 512); 16]) && admitted(2048, 2048, 8, &[(0, 16); 8]) && admitted(1, 512, 1, &[(16, 32)]));
    }
    #[test]
    fn unproven_or_unaligned_batches_keep_the_packed_kernel() {
        let ok = [(256u64, 512u64); 16];
        assert!(!admitted(0, 4096, 16, &ok) && !admitted(2049, 4096, 16, &ok) && !admitted(1024, 256, 16, &ok) && !admitted(1024, 4608, 16, &ok[..1]));
        assert!(!admitted(1024, 8704, 16, &ok) && !admitted(1024, 4096, 17, &[(256, 512); 17]) && !admitted(1024, 4096, 0, &[]));
        assert!(!admitted(1024, 4096, 2, &[(256, 512), (8, 512)]) && !admitted(1024, 4096, 2, &[(256, 512), (256, 520)]));
    }
    #[test]
    fn tile_geometry_matches_the_kernel_layout() {
        assert_eq!(TILE_THREADS, 160);
        assert_eq!(TILE_SHARED, 24064);
        assert!(TILE_SHARED <= 48 * 1024 && TILE_THREADS.is_multiple_of(32));
        assert_eq!(tile_blocks(1024, 16), Some(1024));
        assert_eq!(tile_blocks(2048, 8), Some(1024));
        assert_eq!(tile_blocks(17, 3), Some(6));
        assert_eq!(tile_blocks(1, 1), Some(1));
    }
    #[test]
    fn grid_counts_row_tiles_per_job() {
        assert_eq!(blocks(1024, 16), Some(256));
        assert_eq!(blocks(2048, 8), Some(256));
        assert_eq!(blocks(65, 3), Some(6));
        assert_eq!(blocks(64, 1), Some(1));
    }
    #[test]
    fn production_keeps_original_arithmetic_outside_the_reference_envelope() {
        use crate::expert_ops::{Shape,Op};
        let s=Shape {op:Op::NvProject,rows:1024,cols:4096,block_rows:0,block_cols:0,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0};
        assert!(projection_admitted(s,16,&[(256,512);16]));
        assert!(projection_admitted(Shape {rows:2048,cols:2048,..s},8,&[(256,512);8]));
        for (rows,cols,jobs) in [(258,4096,1),(1,4608,1),(1,512,3),(2,80,1)] {
            assert!(!projection_admitted(Shape {rows,cols,..s},jobs,&vec![(256,512);jobs as usize]));
        }
        assert!(!projection_admitted(Shape {op:Op::Fp8Serial,..s},16,&[(256,512);16]));
        assert!(!projection_admitted(s,16,&[(264,512);16]));
    }
}
