//! NVFP4 prompt rows in shared-memory tiles. Same arithmetic and order as `fp8_rows::nv_row!` (the
//! `expert_group` NvRows path): every output `[row][token]` is ONE ascending FP32 chain over the
//! columns, 16-column block by block, each product `((value * activation) * fold)` rounded in two
//! steps, `fold = (e4m3(scale) * block_activation_scale) * (global * activation)`, no fused multiply-add,
//! the same FP4 values (read from the same decode table) and the same BF16 round at the end.
//!
//! A block takes 256 weight rows and 8 tokens of one expert part; its 128 threads own 4 rows x 4 tokens each
//! (16 independent chains). Columns pass through shared memory in steps of 64 (four 16-column blocks) in
//! ascending order, so a block needs little shared memory and more blocks share an SM. One code byte (two
//! columns of a row) is decoded by one 8-byte read of the (low, high) pair table that `decode_lut` holds.
//! Rows and tokens beyond the part are skipped, never padded into a chain. Lane nv-wide: the same output
//! words as the earlier 128-row form on every byte of the L0 family (benchmarks/2026-11-10/glm-speed-2/
//! prompt-speed/nv-wide/).
//!
//! Launch: entry `nv_rows_tile`, the same `GroupArguments` and table as `expert_group`, grid
//! (ceil(max_tokens/8), ceil(rows/256), parts), 128 threads, no dynamic shared memory. All parts of the
//! launch share rows and cols (checked). Needs op NvRows, cols % 128 == 0, weight and input 16-byte
//! aligned, scales 4-byte aligned, decode table 8-byte aligned; else keep `expert_group`.
use crate::expert_ops::{GroupArguments,GroupEntry,Op};

#[no_mangle]
pub extern "ptx-kernel" fn nv_rows_tile(p:GroupArguments) {
    const BR:usize=256;const BT:usize=8;const NB:usize=4;const THREADS:usize=128;   // rows, tokens and 16-column blocks per step
    const LANES:usize=BT/4;                                                             // threads sharing one row group
    const WP:usize=48;const SP:usize=4;const XB:usize=80;const XP:usize=NB*XB+16;  // weight row, scale row, block and token pitches (bytes)
    const STAGE:usize=BR*WP+BR*SP+BT*XP;
    const TABLES:usize=2*STAGE;const SHARED:usize=TABLES+2048+1024;                 // two stages, the pair table, the scale values
    // SAFETY: The host checks every part (shape, raw owners, input span) and keeps all owners and the table alive
    // through stream completion, as for expert_group. Weights, scales, inputs and the decode table are immutable. Each
    // (part, row, token) output word has exactly one writing thread; the output is private and initialized earlier on
    // the stream, and its index first + row * tokens + token is inside the checked units. The kernel traps unless the
    // block has 128 threads, every part is NvRows with the same rows and cols (a multiple of 128), and the owners are
    // aligned as stated above, so every copy reads whole 32-byte row pieces below `rows` and `cols`, whole 4-byte scale
    // pieces, and token rows below `tokens`. Shared reads and writes stay inside the block's static array (two stages
    // and two tables); a pair-table index is one byte. All 128 threads reach every barrier; loop bounds depend only on
    // the part.
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
        if e.shape.op!=Op::NvRows || cols%128!=0 || rows!=first_part.shape.rows as usize || cols!=first_part.shape.cols as usize
            || (e.weight|(p.stride as u64*4))&15!=0 || e.scales&3!=0 || p.stride as usize!=cols/16*17 {asm!("trap;");}
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
        let (row_bytes,scale_row,stride)=(cols/2,cols/16,p.stride as usize);
        let chunks=cols/64;
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE) as u32;
            // Weights: 32 bytes of each of 256 rows.
            let mut item=tid;
            while item<BR*2 {let (r,q)=(item/2,item%2);let row=row_tile*BR+r;
                if row<rows {crate::ptx::copy16(stage+(r*WP+q*16) as u32,weights.add(row*row_bytes+c*32+q*16).cast());}
                item+=THREADS;}
            // Scales: 4 bytes of each row.
            let mut r=tid;
            while r<BR {if row_tile*BR+r<rows {crate::ptx::copy4(stage+(BR*WP+r*SP) as u32,scales.add((row_tile*BR+r)*scale_row+c*NB).cast());}r+=THREADS;}
            // Activations: 4 blocks of 17 words per token, into 16-byte aligned 20-word slots.
            let mut word=tid;
            while word<BT*NB*17 {let (t,rest)=(word/(NB*17),word%(NB*17));let token=token_tile*BT+t;
                let (b,j)=(rest/17,rest%17);
                if token<m {crate::ptx::copy4(stage+(BR*WP+BR*SP+t*XP+b*XB+j*4) as u32,input.add(token*stride+c*NB*17+rest).cast());}
                word+=THREADS;}
        };
        let (tx,ty)=(tid%LANES,tid/LANES);                                   // rows 4*ty .. 4*ty+3; tokens tx + LANES*j
        let global=mul!(e.shape.global,e.shape.activation);
        let mut acc=[[0.0f32;4];4];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE) as u32;
            for b in 0..NB {
                let mut fold=[[0.0f32;4];4];let mut words=[[0u32;2];4];
                let mut a16=[0.0f32;4];
                rep4!(j,{a16[j]=crate::ptx::load(stage+(BR*WP+BR*SP+(tx+LANES*j)*XP+b*XB+64) as u32);});
                rep4!(i,{
                    let row=ty*4+i;
                    let scale_byte=crate::ptx::load_u8(stage+(BR*WP+row*SP+b) as u32);
                    let value=crate::ptx::load(scale_values+scale_byte*4);
                    rep4!(j,{fold[i][j]=mul!(mul!(value,a16[j]),global);});
                    words[i]=crate::ptx::load2(stage+(row*WP+b*8) as u32);
                });
                rep4!(q,{
                    let mut a=[[0.0f32;4];4];
                    rep4!(j,{let v=crate::ptx::load4(stage+(BR*WP+BR*SP+(tx+LANES*j)*XP+b*XB+q*16) as u32);rep4!(kk,{a[j][kk]=f32::from_bits(v[kk]);});});
                    // Columns 4q+2h and 4q+2h+1 share one code byte (low nibble = the even column); one pair read decodes both.
                    rep2!(h,{
                        let shift=(((q%2)*2+h)*8) as u32;
                        rep4!(i,{
                            let byte=(words[i][q/2]>>shift)&255;
                            let pair=crate::ptx::load2(pairs+byte*8);
                            rep2!(o,{
                                let value=f32::from_bits(pair[o]);
                                rep4!(j,{let product=mul!(mul!(value,a[j][2*h+o]),fold[i][j]);add!(acc[i][j],product);});
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
                    let token=token_tile*BT+tx+LANES*j;
                    if token<m {*(p.output as *mut f32).add(e.first as usize+row*m+token)=crate::expert_ops::round(acc[i][j]);}
                });
            }
        });
    }
}
