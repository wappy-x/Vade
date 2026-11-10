//! Transpose by 4 x 4 tiles. Each thread copies 16 words from four input rows to four output
//! rows, so a warp stays on a few cache lines. A copy does no arithmetic, so the bits are the
//! input's. The host admits only whole tiles of aligned operands.
use crate::prompt_ops::{Arguments,Op};

#[no_mangle]
pub extern "ptx-kernel" fn transpose_tiles(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with
    // same-context contiguous owners that live through stream completion. The input is
    // immutable and the output is private. rows and cols are multiples of 4 and both operands
    // are 16-byte aligned, or the thread traps. `thread<units` bounds the tile, so each 16-byte
    // read stays inside its input row and each 16-byte write inside its output row, and each
    // output word has one writer.
    unsafe {
        use core::arch::asm;
        let thread=crate::index();if thread as u64>=p.units {return;}
        let s=p.shape;let (r,k)=(s.rows as usize,s.cols as usize);
        if s.op!=Op::Transpose || r==0 || k==0 || r%4!=0 || k%4!=0 || (p.input[0]|p.output)&15!=0 {asm!("trap;");}
        let (row,col)=(thread%(r/4)*4,thread/(r/4)*4);
        let (input,output)=(p.input[0] as *const f32,p.output as *mut f32);
        let mut w=[[0u32;4];4];let mut i=0;
        while i<4 {
            let (a,b,c,d):(u32,u32,u32,u32);
            asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)input.add((row+i)*k+col).cast::<u8>(),options(nostack,readonly));
            // With `first` set the exchange is checked here: a non-finite word traps.
            if s.first!=0 && (a&0x7f80_0000==0x7f80_0000 || b&0x7f80_0000==0x7f80_0000 || c&0x7f80_0000==0x7f80_0000 || d&0x7f80_0000==0x7f80_0000) {asm!("trap;");}
            w[i]=[a,b,c,d];i+=1;
        }
        let mut j=0;
        while j<4 {
            asm!("st.global.v4.u32 [{p}], {{{a},{b},{c},{d}}};",a=in(reg32)w[0][j],b=in(reg32)w[1][j],c=in(reg32)w[2][j],d=in(reg32)w[3][j],p=in(reg64)output.add((col+j)*r+row).cast::<u8>(),options(nostack));
            j+=1;
        }
    }
}

/// Transpose through a 32 x 32 shared-memory tile: the same bits as `transpose_tiles` and the generic kernel (a copy does no
/// arithmetic), but a warp reads 32 consecutive words of an input row and writes 32 consecutive words of an output row. The 4 x 4
/// tile kernel reads and writes 16-byte pieces four rows apart: half of every memory sector it touches is wasted, which is why a
/// 33.5 MB exchange upload took 0.92 ms (byte floor 0.3 ms).
///
/// Launch: 256 threads, grid (ceil(cols/32), ceil(rows/32)); the loops stride by the grid, so any grid is correct.
#[no_mangle]
pub extern "ptx-kernel" fn transpose_smem(p:Arguments) {
    // SAFETY: The host (backend/prompt_ops.rs, prompt_admission::transpose_admitted) validated Shape::lengths: the input holds
    // rows * cols words, the output rows * cols words, same-context contiguous owners that live through stream completion, the
    // output private and disjoint from the input. The block must have 256 threads (else it traps). Tile (rt, ct) covers input
    // rows [32 rt, 32 rt + 32) and columns [32 ct, 32 ct + 32): a thread reads only words with row < rows and column < cols,
    // so every read is inside the input, and writes the word at output row (column) and output column (row) with the same two
    // guards, so every write is inside the output and each output word has exactly one writer (its tile, its thread). Shared
    // memory is this kernel's own 4,224 bytes (32 rows of 33 words); every shared read of a tile follows the barrier after
    // the writes of that tile, and the barrier at the end of the tile keeps the next tile's writes behind the reads. Words of
    // the tile outside the matrix are never written and never read (the store guard is the transpose of the load guard). On every
    // successful launch all 256 threads reach both barriers: the tile loops depend only on the shape and the grid (a launch that traps on a
    // non-finite input word with `first` set, as in the generic kernel, is aborted, not a barrier path). The host launch is block (256,1,1) and
    // grid (ceil(cols/32), ceil(rows/32), 1), no dynamic shared memory (driver.rs fixes block y and z, backend.rs grid z, to one); the kernel itself
    // checks block x only, which is enough for this caller and not for an arbitrary multidimensional block.
    unsafe {
        use core::arch::asm;
        let s=p.shape;let (r,k)=(s.rows as usize,s.cols as usize);
        if s.op!=Op::Transpose || r==0 || k==0 || core::arch::nvptx::_block_dim_x()!=256 {core::arch::nvptx::trap();}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (tx,ty)=(tid%32,tid/32);
        let smem=crate::ptx::shared::<4224>();
        let (input,output)=(p.input[0] as *const u32,p.output as *mut u32);
        let check=s.first!=0;
        let (row_tiles,col_tiles)=(r.div_ceil(32),k.div_ceil(32));
        let mut rt=core::arch::nvptx::_block_idx_y() as usize;
        while rt<row_tiles {
            let mut ct=core::arch::nvptx::_block_idx_x() as usize;
            while ct<col_tiles {
                let mut j=0usize;
                while j<4 {
                    let (row,col)=(rt*32+ty+8*j,ct*32+tx);
                    if row<r && col<k {
                        let v=*input.add(row*k+col);
                        if check && v&0x7f80_0000==0x7f80_0000 {core::arch::nvptx::trap();}
                        asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(smem+(((ty+8*j)*33+tx)*4) as u32),v=in(reg32)v);
                    }
                    j+=1;
                }
                crate::ptx::sync();
                j=0;
                while j<4 {
                    let (col,row)=(ct*32+ty+8*j,rt*32+tx);
                    if row<r && col<k {
                        let v:u32;asm!("ld.shared.u32 {v}, [{a}];",v=out(reg32)v,a=in(reg32)(smem+((tx*33+ty+8*j)*4) as u32));
                        *output.add(col*r+row)=v;
                    }
                    j+=1;
                }
                crate::ptx::sync();
                ct+=core::arch::nvptx::_grid_dim_x() as usize;
            }
            rt+=core::arch::nvptx::_grid_dim_y() as usize;
        }
    }
}
