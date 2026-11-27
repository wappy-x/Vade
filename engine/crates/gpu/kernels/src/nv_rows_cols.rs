//! NVFP4 prompt rows over COLUMN-TILE weights (`expert_ops::NV_TILE_ROWS`), in shared-memory tiles. Same arithmetic
//! and order as `nv_rows_tile` and as the `expert_group` NvRows path: every output `[row][token]` is ONE ascending
//! FP32 chain over the columns, 16-column block by block, each product `((value * activation) * fold)` rounded in two
//! steps, `fold = (e4m3(scale) * block_activation_scale) * (global * activation)`, no fused multiply-add, the same
//! FP4 values (read from the same decode table) and the same BF16 round at the end. Output words equal `nv_rows_tile`
//! on the same logical matrix.
//!
//! This is `nv_rows_tile` (the 256-row form of lane nv-wide) with other addresses. A block takes two 128-row tiles
//! (256 rows) and 8 tokens of one expert part; its 128 threads own 4 rows x 4 tokens each (16 independent chains).
//! Columns pass through shared memory in steps of 64 in ascending order: a step is, for each of the two tiles, its 64
//! columns of 64 bytes (one contiguous 4 KB run of the column-tile layout) and its 4 scale blocks of 128 bytes. A byte
//! of a column holds two neighbouring rows; one 8-byte read of the (low, high) pair table that `decode_lut` holds
//! decodes both. No column is left out here: a prompt column is zero only for some tokens.
//!
//! Launch: entry `nv_rows_cols`, the same `GroupArguments` and table as `expert_group`, grid
//! (ceil(max_tokens/8), ceil(rows/256), parts), 128 threads, no dynamic shared memory. All parts of the launch share rows
//! and cols (checked). Needs op NvRows on column tiles, rows % 128 == 0, cols % 128 == 0, weight, scales and input
//! 16-byte aligned, decode table 8-byte aligned; else keep `expert_group`.
use crate::expert_ops::{GroupArguments,GroupEntry,Op};

const BR:usize=256;const BT:usize=8;const NB:usize=4;const THREADS:usize=128;               // rows, tokens and 16-column blocks per step
const TILE:usize=128;const CP:usize=64;const SB:usize=128;const XB:usize=80;const XP:usize=NB*XB+16; // tile rows; column, scale block, activation block and token pitches (bytes)
const COLS:usize=NB*16;const WEIGHTS:usize=2*COLS*CP;const SCALES:usize=2*NB*SB;              // one step: two tiles of 64 columns and 4 scale blocks
const STAGE:usize=WEIGHTS+SCALES+BT*XP;
const TABLES:usize=2*STAGE;const SHARED:usize=TABLES+2048+1024;                                // two stages, the pair table, the scale values

#[no_mangle]
pub extern "ptx-kernel" fn nv_rows_cols(p:GroupArguments) {
    // SAFETY: The host checks every part (shape, raw owners, input span) and keeps all owners and the table
    // alive through stream completion, as for expert_group. Weights, scales, inputs and the decode table
    // are immutable. Each (part, row, token) output word has exactly one writing thread; the output is
    // private and initialized earlier on the stream, and its index first + row * tokens + token is
    // inside the checked units. The kernel traps unless the block has 128 threads, every part is NvRows on
    // column tiles with the same rows (a multiple of 128) and cols (a multiple of 128), and the owners are
    // aligned as stated above, so every 16-byte copy reads whole columns and whole scale blocks of the block's
    // one or two tiles below `rows` and below `cols`, and token rows below `tokens`. Rows of an absent tile
    // read stale shared words and are never stored. Shared reads and writes stay inside
    // the block's static array (two stages and two tables); a pair-table index is one byte. All 128 threads reach
    // every barrier; loop bounds depend only on the part.
    unsafe {
        use core::arch::asm;
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        // Straight-line copies, so every array index below is a constant and the arrays stay in registers.
        macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
        macro_rules! rep2 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}}};}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        if core::arch::nvptx::_block_dim_x()!=THREADS as u32 || p.count==0 || p.words!=1 || p.lut&7!=0 || p.input&15!=0 {asm!("trap;");}
        let part=core::arch::nvptx::_block_idx_z() as usize;if part>=p.count as usize {return;}
        let table=core::slice::from_raw_parts(p.table as *const GroupEntry,p.count as usize);
        let first_part=&table[0];let e=&table[part];
        let (rows,cols,m)=(e.shape.rows as usize,e.shape.cols as usize,e.shape.tokens as usize);
        if e.shape.op!=Op::NvRows || !crate::expert_ops::nv_tiled(&e.shape) || rows%TILE!=0 || cols%128!=0 || rows!=first_part.shape.rows as usize || cols!=first_part.shape.cols as usize || (e.weight|e.scales|(p.stride as u64*4))&15!=0 || p.stride as usize!=cols/16*17 {asm!("trap;");}
        let (token_tile,row_tile)=(core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize);
        if row_tile*BR>=rows || token_tile*BT>=m {return;}
        let smem=crate::ptx::shared::<SHARED>();
        let lut=p.lut as *const f32;
        // The 256 (low, high) code pairs and the 256 scale values, from the same table as every other NVFP4 row body.
        let (pairs,scale_values)=(smem+TABLES as u32,smem+(TABLES+2048) as u32);
        let mut at=tid;while at<512 {crate::ptx::store(pairs+(at*4) as u32,*lut.add(at));at+=THREADS;}
        let mut at=tid;while at<256 {crate::ptx::store(scale_values+(at*4) as u32,*lut.add(512+at));at+=THREADS;}
        let weights=e.weight as *const u8;let scales=e.scales as *const u8;
        let input=(p.input as *const f32).add(e.first_row as usize*p.stride as usize);
        // The block's two tiles are tiles 2*row_tile and 2*row_tile+1 of the matrix.
        let (tile_codes,tile_scales,stride)=(weights.add(2*row_tile*cols*CP),scales.add(2*row_tile*(cols/16)*SB),p.stride as usize);
        let chunks=cols/COLS;
        // The last block of a matrix with an odd number of 128-row tiles has one tile; nothing of the absent tile is read.
        let tiles=if (2*row_tile+1)*TILE<rows {2} else {1};
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE) as u32;
            // Weights: for each tile, its 64 columns of this step, 64 bytes each, one contiguous 4 KB run.
            let mut item=tid;
            while item<tiles*COLS*4 {let (t,rest)=(item/(COLS*4),item%(COLS*4));
                crate::ptx::copy16(stage+(t*COLS*CP+rest*16) as u32,tile_codes.add(t*cols*CP+c*COLS*CP+rest*16).cast());item+=THREADS;}
            // Scales: for each tile, the 4 blocks of this step, 128 bytes each (one byte a row).
            if tid<tiles*NB*SB/16 {let (t,rest)=(tid/(NB*SB/16),tid%(NB*SB/16));
                crate::ptx::copy16(stage+(WEIGHTS+t*NB*SB+rest*16) as u32,tile_scales.add(t*(cols/16)*SB+c*NB*SB+rest*16).cast());}
            // Activations: 4 blocks of 17 words per token, into 16-byte aligned 20-word slots.
            let mut word=tid;
            while word<BT*NB*17 {let (t,rest)=(word/(NB*17),word%(NB*17));let (b,j)=(rest/17,rest%17);let token=token_tile*BT+t;
                if token<m {crate::ptx::copy4(stage+(WEIGHTS+SCALES+t*XP+b*XB+j*4) as u32,input.add(token*stride+c*NB*17+rest).cast());}
                word+=THREADS;}
        };
        let (tx,ty)=(tid%2,tid/2);                                           // rows 4*ty .. 4*ty+3; tokens tx, tx+2, tx+4, tx+6
        let (tile,within)=((4*ty)/TILE,(4*ty)%TILE);                          // the tile of those rows and the first row inside it
        let global=mul!(e.shape.global,e.shape.activation);
        let mut acc=[[0.0f32;4];4];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE) as u32;
            for b in 0..NB {
                let mut fold=[[0.0f32;4];4];
                let mut a16=[0.0f32;4];
                rep4!(j,{a16[j]=crate::ptx::load(stage+(WEIGHTS+SCALES+(tx+2*j)*XP+b*XB+64) as u32);});
                rep4!(i,{
                    let scale_byte=crate::ptx::load_u8(stage+(WEIGHTS+tile*NB*SB+b*SB+within+i) as u32);
                    let value=crate::ptx::load(scale_values+scale_byte*4);
                    rep4!(j,{fold[i][j]=mul!(mul!(value,a16[j]),global);});
                });
                rep4!(q,{
                    let mut a=[[0.0f32;4];4];
                    rep4!(j,{let v=crate::ptx::load4(stage+(WEIGHTS+SCALES+(tx+2*j)*XP+b*XB+q*16) as u32);rep4!(kk,{a[j][kk]=f32::from_bits(v[kk]);});});
                    rep4!(kk,{
                        // Column 16*b + 4*q + kk: two bytes of the staged column hold this thread's four rows; each byte is
                        // one (even row, odd row) pair of the pair table.
                        let packed=crate::ptx::load_u16(stage+((tile*COLS+b*16+q*4+kk)*CP+within/2) as u32);
                        rep2!(h,{
                            let pair=crate::ptx::load2(pairs+((packed>>(8*h as u32))&255)*8);
                            rep2!(o,{
                                let i=2*h+o;let value=f32::from_bits(pair[o]);
                                rep4!(j,{let product=mul!(mul!(value,a[j][kk]),fold[i][j]);add!(acc[i][j],product);});
                            });
                        });
                    });
                });
            }
            crate::ptx::sync();
            c+=1;
        }
        rep4!(i,{
            let row=row_tile*BR+ty*4+i;
            if row<rows {
                rep4!(j,{
                    let token=token_tile*BT+tx+2*j;
                    if token<m {*(p.output as *mut f32).add(e.first as usize+row*m+token)=crate::expert_ops::round(acc[i][j]);}
                });
            }
        });
    }
}
