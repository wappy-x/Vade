//! Attention scores in shared-memory tiles, with lane K's plain tile method. For one head, a
//! block takes 64 key rows and 64 tokens; each of its 256 threads owns 4 rows x 4 tokens, 16
//! independent chains. Every output keeps one ascending FP32 chain over the key columns, each
//! product rounded before its add, then the same two rounds as the row kernel. All pairs of a
//! tile are computed. A host map names the slot of each selected pair; the others are dropped.
use crate::prompt_ops::{Arguments,Op,root,round};

#[no_mangle]
pub extern "ptx-kernel" fn scores_rows_tile(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with
    // same-context contiguous owners that live through stream completion. Queries and keys are
    // immutable and 16-byte aligned (checked here, else the block traps); dim and taps are
    // multiples of 4, so every 16-byte copy reads four whole key columns inside its row, and
    // only rows below `rows` and tokens below `tokens` are copied. Map entries are data: a slot
    // must be live for its token and hold this row, or the block traps, so writes stay inside
    // the private output, and a correct map has one writer per word. Every shared read and
    // write stays in the block's own two-stage static array. All 256 threads reach every
    // barrier; the loop bounds depend only on the shape.
    unsafe {
        const BR:usize=64;const BT:usize=64;const KC:usize=32;const SK:usize=36;const STAGE:usize=(BR+BT)*SK;
        let s=p.shape;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::ScoresMap || s.dim==0 || s.dim%4!=0 || s.taps%4!=0 || (p.input[0]|p.input[1])&15!=0 || threads!=256 {core::arch::asm!("trap;");}
        let (m,h,d,t,rows,slots,cols)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.slots as usize,s.cols as usize);
        let row_words=h*(d+t);let words=p.words as usize;
        let token_tiles=m.div_ceil(BT);let tiles=token_tiles*rows.div_ceil(BR);
        let block=core::arch::nvptx::_block_idx_x() as usize;
        let (head,tile)=(block/tiles,block%tiles);let (tile_t,tile_r)=(tile%token_tiles,tile/token_tiles);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let smem=crate::ptx::shared::<{2*STAGE*4}>();
        let keys=(p.input[1] as *const f32).add(head*(d+t));let queries=(p.input[0] as *const f32).add(head*d);
        let chunks=d.div_ceil(KC);
        // Copy chunk `c` of this block's 64 key rows and 64 queries into stage `c & 1`.
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE*4) as u32;let first=c*KC;
            let mut item=tid;
            while item<(BR+BT)*8 {
                let (which,row,quad)=(item/(BR*8),item%(BR*8)/8,item%8);let column=first+quad*4;
                if column<d {
                    let dst=stage+((which*BR+row)*SK*4+quad*16) as u32;
                    if which==0 {
                        let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(dst,keys.add(r*row_words+column).cast());}
                    } else {
                        let token=tile_t*BT+row;if token<m {crate::ptx::copy16(dst,queries.add(token*h*d+column).cast());}
                    }
                }
                item+=256;
            }
        };
        let mut acc=[[0.0f32;4];4];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE*4) as u32;let valid=(d-c*KC).min(KC);
            let mut q=0;
            while q<valid/4 {
                let mut w=[[0u32;4];4];let mut x=[[0u32;4];4];
                for i in 0..4 {w[i]=crate::ptx::load4(stage+(((ty*4+i)*SK+q*4)*4) as u32);}
                for j in 0..4 {x[j]=crate::ptx::load4(stage+(((BR+tx+16*j)*SK+q*4)*4) as u32);}
                for e in 0..4 {
                    for i in 0..4 {for j in 0..4 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                }
                q+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        let scale=1.0/root(d as f32);let map=p.input[3] as *const f32;let selection=p.input[2] as *const f32;
        for i in 0..4 {
            let row=tile_r*BR+ty*4+i;if row>=rows {continue;}
            for j in 0..4 {
                let token=tile_t*BT+tx+16*j;if token>=m {continue;}
                let slot=*map.add(row*m+token);if slot==-1.0 {continue;}let slot=slot as usize;
                let selected=selection.add(token*(slots+1));
                let count=*selected.add(slots) as usize;
                if count>cols.min(slots) || slot>=count || *selected.add(slot) as usize!=row {panic!();}
                *(p.output as *mut f32).add((token*h+head)*words+slot)=round(round(acc[i][j])*scale);
            }
        }
    }
}

/// The same tile over expanded rows packed two BF16 patterns a word. Key staging widens each packed
/// half into the identical shared layout, so every chain and output bit equals `scores_rows_tile` on
/// the unpacked rows. Queries still stage with copy16; keys stage by register (one 8-byte load widens
/// to the same four shared words), because cp.async cannot widen.
#[no_mangle]
pub extern "ptx-kernel" fn scores_rows_tile_packed(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and admits only packed_admitted shapes (dim and taps
    // 256, listed heads, 16-byte aligned queries and packed rows). Each packed head segment is 256
    // words, so every 8-byte key load (four columns, two packed words) stays inside its row and stays
    // 8-byte aligned; rows below `rows` and tokens below `tokens` only. The widened halves are the
    // exact f32 words the unpacked row held (every expanded word is BF16-rounded at production).
    // Map entries are data and checked as in scores_rows_tile. Shared reads and writes stay in the
    // block's own two-stage array; all 256 threads reach every barrier; loop bounds follow the shape.
    unsafe {
        const BR:usize=64;const BT:usize=64;const KC:usize=32;const SK:usize=36;const STAGE:usize=(BR+BT)*SK;
        let s=p.shape;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::ScoresMapPacked || s.dim!=256 || s.taps!=256 || (p.input[0]|p.input[1])&15!=0 || threads!=256 {core::arch::asm!("trap;");}
        let (m,h,d,t,rows,slots,cols)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.slots as usize,s.cols as usize);
        let packed_row_words=h*(d+t)/2;let words=p.words as usize;
        let token_tiles=m.div_ceil(BT);let tiles=token_tiles*rows.div_ceil(BR);
        let block=core::arch::nvptx::_block_idx_x() as usize;
        let (head,tile)=(block/tiles,block%tiles);let (tile_t,tile_r)=(tile%token_tiles,tile/token_tiles);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let smem=crate::ptx::shared::<{2*STAGE*4}>();
        let keys=(p.input[1] as *const u32).add(head*(d+t)/2);let queries=(p.input[0] as *const f32).add(head*d);
        let chunks=d.div_ceil(KC);
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE*4) as u32;let first=c*KC;
            let mut item=tid;
            while item<(BR+BT)*8 {
                let (which,row,quad)=(item/(BR*8),item%(BR*8)/8,item%8);let column=first+quad*4;
                if column<d {
                    let dst=stage+((which*BR+row)*SK*4+quad*16) as u32;
                    if which==0 {
                        let r=tile_r*BR+row;
                        if r<rows {
                            let (w0,w1):(u32,u32);
                            core::arch::asm!("ld.global.nc.v2.u32 {{{a},{b}}}, [{pointer}];",a=out(reg32)w0,b=out(reg32)w1,pointer=in(reg64)keys.add(r*packed_row_words+column/2),options(nostack,readonly));
                            core::arch::asm!("st.shared.v4.b32 [{dst}], {{{a},{b},{c},{e}}};",dst=in(reg32)dst,a=in(reg32)(w0<<16),b=in(reg32)(w0&0xffff_0000),c=in(reg32)(w1<<16),e=in(reg32)(w1&0xffff_0000),options(nostack));
                        }
                    } else {
                        let token=tile_t*BT+row;if token<m {crate::ptx::copy16(dst,queries.add(token*h*d+column).cast());}
                    }
                }
                item+=256;
            }
        };
        let mut acc=[[0.0f32;4];4];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE*4) as u32;let valid=(d-c*KC).min(KC);
            let mut q=0;
            while q<valid/4 {
                let mut w=[[0u32;4];4];let mut x=[[0u32;4];4];
                for i in 0..4 {w[i]=crate::ptx::load4(stage+(((ty*4+i)*SK+q*4)*4) as u32);}
                for j in 0..4 {x[j]=crate::ptx::load4(stage+(((BR+tx+16*j)*SK+q*4)*4) as u32);}
                for e in 0..4 {
                    for i in 0..4 {for j in 0..4 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                }
                q+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        let scale=1.0/root(d as f32);let map=p.input[3] as *const f32;let selection=p.input[2] as *const f32;
        for i in 0..4 {
            let row=tile_r*BR+ty*4+i;if row>=rows {continue;}
            for j in 0..4 {
                let token=tile_t*BT+tx+16*j;if token>=m {continue;}
                let slot=*map.add(row*m+token);if slot==-1.0 {continue;}let slot=slot as usize;
                let selected=selection.add(token*(slots+1));
                let count=*selected.add(slots) as usize;
                if count>cols.min(slots) || slot>=count || *selected.add(slot) as usize!=row {panic!();}
                *(p.output as *mut f32).add((token*h+head)*words+slot)=round(round(acc[i][j])*scale);
            }
        }
    }
}
