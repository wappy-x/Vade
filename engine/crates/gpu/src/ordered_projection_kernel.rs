//! One thread per FP32 output row. Keep the scalar reference operation order.
#[no_mangle]
pub extern "ptx-kernel" fn project_f32_ordered(weights: *const f32,input: *const f32,output: *mut f32,rows: u32,cols: u32) {resident_project_f32_ordered(weights,input,output,rows,cols)}
/// Device check only (kernel-bench --plain-probe): the same body, but the block's shared memory is filled with NaN
/// before the tile runs, so any read of a slot that no copy wrote turns a valid output into NaN.
#[no_mangle]
pub extern "ptx-kernel" fn project_f32_ordered_poisoned(weights: *const f32,input: *const f32,output: *mut f32,rows: u32,cols: u32) {project_body::<true>(weights,input,output,rows,cols)}
#[inline(always)]
pub(crate) fn resident_project_f32_ordered(weights:*const f32,input:*const f32,output:*mut f32,rows:u32,cols:u32) {project_body::<false>(weights,input,output,rows,cols)}
#[inline(always)]
fn project_body<const POISON:bool>(weights:*const f32,input:*const f32,output:*mut f32,rows:u32,cols:u32) {
    // SAFETY: The backend checks contiguous FP32 extents, context, alignment,
    // address overflow, and disjoint output storage before this launch.
    // The owners stay live through completion. Each valid thread owns one row.
    // The row check bounds every access. Shape checks bound row offset math.
    // Schedule by shape only (plain_admission): a FOLD_THREADS-thread block inside the envelope takes the fold, where
    // every thread of a block whose first row is below `rows` enters rows_fold and only the folding lanes of live rows
    // store (the first-row test is the same on all threads of a block, so the named barriers see the whole block);
    // a 32- or 128-thread block of whole warps inside the swept
    // envelope takes the warp tile, where a warp whose first row is below `rows` runs every lane through the same
    // code and only lanes with a valid row compute and store; other vector shapes inside the envelope take the
    // register schedule; the rest keep the original loop. The shared block is the one this kernel declares and each
    // warp uses only its own part of it. All schedules compute the same bytes.
    unsafe {
        let (rows,cols)=(rows as usize,cols as usize);
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        // ONE shared declaration for both shared schedules: ptxas adds separate declarations of one entry together.
        let smem=crate::ptx::shared::<{crate::plain_warp::BLOCK_BYTES}>();
        if crate::plain_admission::fold_admitted(rows,cols,weights as usize,input as usize,threads) {
            // Fold blocks (plain_admission::plain_geometry for few long rows): block b owns rows [b*FOLD_ROWS, b*FOLD_ROWS+FOLD_ROWS);
            // every thread of the block enters rows_fold, and only lane r < FOLD_ROWS of warp 0 stores, its own live row.
            if POISON {
                let mut at=tid;while at<crate::plain_warp::FOLD_BYTES/4 {core::arch::asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(smem+(at*4) as u32),v=in(reg32)0x7fc0_0000u32);at+=threads;}
                core::arch::asm!("bar.sync 0;");
            }
            let first=core::arch::nvptx::_block_idx_x() as usize*crate::plain_admission::FOLD_ROWS;
            if first<rows {
                let sum=crate::plain_warp::rows_fold(weights,input,cols,first,rows,smem);
                if tid<crate::plain_admission::FOLD_ROWS && first+tid<rows {*output.add(first+tid)=sum;}
            }
        } else if crate::plain_admission::tile_admitted(rows,cols,weights as usize,input as usize,threads) {
            if POISON {
                let mut at=tid;while at<crate::plain_warp::BLOCK_BYTES/4 {core::arch::asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(smem+(at*4) as u32),v=in(reg32)0x7fc0_0000u32);at+=threads;}
                core::arch::asm!("bar.sync 0;");
            }
            let first=core::arch::nvptx::_block_idx_x() as usize*threads+tid/32*32;
            if first<rows {
                let sum=if threads==crate::plain_admission::NARROW_THREADS {
                    crate::plain_warp::rows_warp::<{crate::plain_warp::NS_ALONE}>(weights,input,cols,first,rows,smem)
                } else {
                    crate::plain_warp::rows_warp::<{crate::plain_warp::NS_BLOCK}>(weights,input,cols,first,rows,smem+(tid/32*crate::plain_warp::WARP_BYTES) as u32)
                };
                if first+tid%32<rows {*output.add(first+tid%32)=sum;}
            }
        } else {
            let row=crate::index();
            if row<rows {
                *output.add(row)=if crate::plain_admission::deep_admitted(rows,cols,weights as usize,input as usize) {
                    crate::plain_warp::row_deep(weights.add(row*cols),input,cols)
                } else {
                    crate::serial_projection::plain_row!(weights.add(row*cols),input,cols)
                };
            }
        }
    }
}
