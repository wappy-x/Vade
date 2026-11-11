//! Data movement of the prompt rows: `Op::PermuteRows` and `Op::ExpertPlaceTile`. Both only copy words, so the
//! bytes are those of the generic `prompt_rows` kernel for every table the host admits. The generic kernel gives
//! one thread to each output word through a body that holds every operation (87 registers, a table load before
//! each data load, a 4-byte store); a group of 256 tokens moves 33 to 67 MB per call there, at 25 to 50 GB/s.
//!
//! `prompt_permute`: one block of 256 threads per output row; the row's source index is read once; the row moves
//! as 16-byte vectors, four in flight per thread.
//!
//! `prompt_place`: the experts' results arrive channel-major per expert group, so reading one output row's
//! channels is a stride of "group rows" words. A block takes 32 output rows that are consecutive in expert order
//! (slots) and 32 channels: it reads along the slots (consecutive words of one group), turns the tile in shared
//! memory, and writes along the channels (128 contiguous bytes of an output row).
use core::arch::asm;
use crate::prompt_ops::{Arguments,Op};

const THREADS:usize=crate::prompt_admission::MOVE_THREADS as usize;

#[no_mangle]
pub extern "ptx-kernel" fn prompt_permute(p:Arguments) {
    // SAFETY: The host (backend/prompt_ops.rs, prompt_admission::permute_admitted) checked Shape::lengths: the source
    // holds `first` rows of `cols` words, the table `rows` words, the output `rows * cols` words, all owners live and
    // disjoint through completion, and source and output 16-byte aligned with `cols` divisible by 4. The block must have
    // THREADS threads (else it traps). A block owns output row `row` (strided by the grid, so any grid is correct):
    // its threads write the vectors `v < cols/4` of that row, each by exactly one thread, so every output word has one
    // writer and no two blocks share a row. The table entry is data: it is read as the generic kernel reads it (a
    // float cast to usize, NaN and negative to 0) and a row at or beyond `first` traps, as the generic slice index
    // does. Every read is inside the source row it names. No shared memory and no barrier.
    unsafe {
        let s=p.shape;
        if s.op!=Op::PermuteRows || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (rows,cols,source_rows)=(s.rows as usize,s.cols as usize,s.first as usize);
        let vectors=cols/4;let tid=core::arch::nvptx::_thread_idx_x() as usize;
        let table=p.input[1] as *const f32;
        let mut row=core::arch::nvptx::_block_idx_x() as usize;
        while row<rows {
            let id=*table.add(row) as usize;
            if id>=source_rows {core::arch::nvptx::trap();}
            let from=(p.input[0] as *const u32).add(id*cols);let to=(p.output as *mut u32).add(row*cols);
            let mut base=0usize;
            while base<vectors {
                let mut held=[[0u32;4];4];
                let mut j=0;
                while j<4 {
                    let v=base+j*THREADS+tid;
                    if v<vectors {
                        let at=from.add(v*4);
                        let (a,b,c,d):(u32,u32,u32,u32);
                        asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{at}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,at=in(reg64)at,options(nostack,readonly));
                        held[j]=[a,b,c,d];
                    }
                    j+=1;
                }
                j=0;
                while j<4 {
                    let v=base+j*THREADS+tid;
                    if v<vectors {
                        let at=to.add(v*4);let h=held[j];
                        asm!("st.global.v4.u32 [{at}], {{{a},{b},{c},{d}}};",at=in(reg64)at,a=in(reg32)h[0],b=in(reg32)h[1],c=in(reg32)h[2],d=in(reg32)h[3],options(nostack));
                    }
                    j+=1;
                }
                base+=4*THREADS;
            }
            row+=core::arch::nvptx::_grid_dim_x() as usize;
        }
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn prompt_place(p:Arguments) {
    // SAFETY: The host (prompt_admission::place_admitted) checked Shape::lengths for Op::ExpertPlaceTile: the exchange
    // holds `slots * taps` words, the table `4 * rows` words (`rows` triples, then the slot to row inverse built by
    // prompt_ops::place_table, a permutation of 0..rows), the output `rows * cols` words with `cols = dim * taps`
    // divisible by 32; owners live and disjoint through completion. The block must have THREADS threads (else it
    // traps). A block owns the output words of 32 slots by 32 channels: (slot tile, channel tile) come from the grid
    // and are strided by it, so any grid is correct. Each output word (row, channel) belongs to exactly one slot and
    // one channel, hence to one tile and one thread; the inverse is a permutation, so no two slots share a row. Table
    // entries are data and are checked before use: a row at or beyond `rows`, a triple whose `earlier + at` is not
    // its slot, or a source index at or beyond `slots * taps` traps (the generic kernel's slice index traps on the
    // last). Shared memory is this kernel's own 4,352 bytes: the 32x33-word tile and 32 row words. Every shared read
    // follows a write: phase 1 writes tile[c][s] for all 32 channels c of every slot s < rows - tile start (thread
    // (s, ty) writes channels ty, ty+8, ty+16, ty+24) and rows[s] (ty = 0), phase 2 reads only those slots, and
    // each tile ends with a barrier before the next tile reuses the shared words. A slot beyond the last row writes
    // and reads nothing. All threads reach every barrier: loop bounds depend only on the shape and the grid.
    unsafe {
        let s=p.shape;
        if s.op!=Op::ExpertPlaceTile || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (rows,cols,d,parts,slots)=(s.rows as usize,s.cols as usize,s.dim as usize,s.taps as usize,s.slots as usize);
        let edge=crate::prompt_admission::TILE as usize;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let tx=tid%32;let ty=tid/32;
        let exchange=p.input[0] as *const u32;let table=p.input[1] as *const f32;let output=p.output as *mut u32;
        let smem=crate::ptx::shared::<4352>();const ROWS:u32=4224;
        let slot_tiles=rows.div_ceil(edge);let channel_tiles=cols/edge;
        let mut st=core::arch::nvptx::_block_idx_x() as usize;
        while st<slot_tiles {
            let mut ct=core::arch::nvptx::_block_idx_y() as usize;
            while ct<channel_tiles {
                let s0=st*edge;let c0=ct*edge;let slot=s0+tx;
                let mut earlier=0usize;let mut tokens=0usize;let mut at=0usize;
                if slot<rows {
                    let row=*table.add(3*rows+slot) as usize;
                    if row>=rows {core::arch::nvptx::trap();}
                    let e=table.add(3*row);
                    earlier=*e as usize;tokens=*e.add(1) as usize;at=*e.add(2) as usize;
                    if earlier+at!=slot {core::arch::nvptx::trap();}
                    if ty==0 {asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(smem+ROWS+(tx as u32)*4),v=in(reg32)row as u32);}
                    let mut j=0;
                    while j<4 {
                        let channel=c0+ty+8*j;
                        let source=(channel/d)*slots+earlier*d+(channel%d)*tokens+at;
                        if source>=slots*parts {core::arch::nvptx::trap();}
                        let v=*exchange.add(source);
                        asm!("st.shared.u32 [{a}], {v};",a=in(reg32)(smem+(((ty+8*j)*33+tx) as u32)*4),v=in(reg32)v);
                        j+=1;
                    }
                }
                asm!("bar.sync 0;");
                let mut j=0;
                while j<4 {
                    let local=ty+8*j;
                    if s0+local<rows {
                        let row:u32;let v:u32;
                        asm!("ld.shared.u32 {r}, [{a}];",r=out(reg32)row,a=in(reg32)(smem+ROWS+(local as u32)*4));
                        asm!("ld.shared.u32 {v}, [{a}];",v=out(reg32)v,a=in(reg32)(smem+((tx*33+local) as u32)*4));
                        *output.add(row as usize*cols+c0+tx)=v;
                    }
                    j+=1;
                }
                asm!("bar.sync 0;");
                ct+=core::arch::nvptx::_grid_dim_y() as usize;
            }
            st+=core::arch::nvptx::_grid_dim_x() as usize;
        }
    }
}
