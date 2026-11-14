//! Load-time permutation of NVFP4 planes from row-major to column tiles (`expert_ops::NV_TILE_ROWS`), on the device.
//! Pure byte moves: every output byte holds the same two 4-bit codes, and every output scale byte the same scale, as
//! the host functions `expert_ops::nv_tile_codes_in_place` and `nv_tile_scales_in_place` write (device check:
//! `resident-decode-check nv-group-rows`; host twin: `expert_ops::tests`).
//! A block takes one 128-row tile and one chunk of 64 columns (4 scale blocks). It stages the 32 code bytes and the 4
//! scale bytes of each of its 128 rows in shared memory with asynchronous copies, waits once, then every thread builds
//! whole output words and stores them: the 64 bytes of a column are 16 words (4 row pairs a word, row 2k in the low
//! nibble of byte k), the 128 bytes of a scale block are 32 words (4 rows a word). Half a warp writes one whole column.
//! Launch: entry `nv_tile_repack`, `TileRepack` arguments, `nv_tile_repack_blocks` blocks (tile-major), 128 threads.
use crate::expert_ops::{TileRepack,NV_REPACK_CODE_STAGE,NV_REPACK_SHARED,NV_REPACK_THREADS,NV_TILE_ROWS};
use crate::expert_ops::{nv_repack_code_slot,nv_repack_code_unit,nv_repack_scale_slot};

#[no_mangle]
pub extern "ptx-kernel" fn nv_tile_repack(p:TileRepack) {
    // SAFETY: The host (`Backend::nv_tile_repack`) checks that rows is a multiple of 128 and cols a multiple of 64 inside
    // the storage rule, that the four planes are different contiguous allocations of rows*cols/2 code bytes and
    // rows*cols/16 scale bytes, 16-byte aligned, and keeps them alive through stream completion; nothing else reads or
    // writes the destination planes before that. A block reads 32 code bytes and 4 scale bytes of each of the 128 rows
    // of its tile at its chunk (inside the source planes) and writes the 64 columns x 64 bytes and the 4 scale blocks x
    // 128 bytes of the same tile and chunk (inside the destination planes); blocks write disjoint ranges, and inside a
    // block every output word has one writing thread (64 x 16 code words over 8 steps, 4 x 32 scale words). Shared
    // addresses stay inside the static stage; a staged byte is written by one copy and read after the copier's wait
    // and the barrier that every thread executes once. The kernel traps on any other launch.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        let tid=_thread_idx_x() as usize;
        let (rows,cols)=(p.rows as usize,p.cols as usize);let tile_rows=NV_TILE_ROWS as usize;
        if _block_dim_x()!=NV_REPACK_THREADS || rows==0 || rows%tile_rows!=0 || cols==0 || cols%64!=0 || (p.source_codes|p.source_scales|p.codes|p.scales)&15!=0 {asm!("trap;");}
        let chunks=cols/64;let block=_block_idx_x() as usize;let (tile,chunk)=(block/chunks,block%chunks);
        if tile>=rows/tile_rows {return;}
        let (half,blocks)=(cols/2,cols/16);
        let stage=crate::ptx::shared::<{NV_REPACK_SHARED as usize}>();let scale_stage=stage+NV_REPACK_CODE_STAGE;
        // Stage: 4 pieces of 8 bytes a row (codes), one word a row (scales).
        let source_codes=(p.source_codes as *const u8).add(tile*tile_rows*half+chunk*32);
        let mut piece=tid;
        while piece<tile_rows*4 {
            let (row,part)=(piece/4,piece%4);
            crate::ptx::copy8(stage+(nv_repack_code_slot(row)+part*8) as u32,source_codes.add(row*half+part*8).cast());
            piece+=NV_REPACK_THREADS as usize;
        }
        crate::ptx::copy4(scale_stage+nv_repack_scale_slot(tid) as u32,(p.source_scales as *const u8).add((tile*tile_rows+tid)*blocks+chunk*4).cast());
        crate::ptx::commit();
        asm!("cp.async.wait_all;");
        crate::ptx::sync();
        // Codes: one output word a step.
        let codes=(p.codes as *mut u8).add((tile*cols+chunk*64)*64);
        let mut step=0;
        while step<8 {
            let (column,word)=nv_repack_code_unit(tid,step);
            let shift=4*(column as u32&1);let base=stage+(nv_repack_code_slot(8*word)+column/2) as u32;
            // Rows 8w .. 8w+7 of this column: the staged rows of one group are 32 bytes apart.
            let b=[crate::ptx::load_u8(base),crate::ptx::load_u8(base+32),crate::ptx::load_u8(base+64),crate::ptx::load_u8(base+96),
                crate::ptx::load_u8(base+128),crate::ptx::load_u8(base+160),crate::ptx::load_u8(base+192),crate::ptx::load_u8(base+224)];
            let pair=|even:u32,odd:u32|((even>>shift)&15)|(((odd>>shift)&15)<<4);
            let value=pair(b[0],b[1])|(pair(b[2],b[3])<<8)|(pair(b[4],b[5])<<16)|(pair(b[6],b[7])<<24);
            *(codes.add(column*64+word*4) as *mut u32)=value;
            step+=1;
        }
        // Scales: thread = (scale block 0..4 of the chunk, word 0..32); rows 4w .. 4w+3.
        let (scale_block,word)=(tid/32,tid%32);
        let at=|row:usize|crate::ptx::load_u8(scale_stage+(nv_repack_scale_slot(row)+scale_block) as u32);
        let value=at(4*word)|(at(4*word+1)<<8)|(at(4*word+2)<<16)|(at(4*word+3)<<24);
        *((p.scales as *mut u8).add((tile*blocks+chunk*4+scale_block)*tile_rows+word*4) as *mut u32)=value;
    }
}
