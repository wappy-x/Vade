//! Block-FP8 prompt rows in shared-memory tiles. Same arithmetic and order as `fp8_rows::row!` (the
//! `expert_group` Fp8Rows path): every output `[row][token]` is ONE ascending FP32 chain over the columns
//! from +0.0; each element adds `(e4m3(w) * scale) * x` made by two separate `mul.rn.f32`, each step one
//! separate `add.rn.f32`, no fused multiply-add; the same hardware E4M3 pair conversion; the same final
//! BF16 round. `scale` is the block scale of the element's 16-column group.
//!
//! A block takes 64 weight rows and 64 tokens of one part; each of its 256 threads owns 4 rows x 4 tokens
//! (16 independent chains), so one decoded and scaled weight feeds 4 tokens and one activation feeds 4
//! rows. Columns pass through shared memory 64 at a time in ascending order (asynchronous 16- and 4-byte
//! copies, double buffered). Rows and tokens beyond the part are skipped, never padded into a chain.
//!
//! Launch: entry `fp8_rows_tile`, the same `GroupArguments` and table as `expert_group`, grid
//! (ceil(max_tokens/64), ceil(rows/64), parts), 256 threads, no dynamic shared memory. Needs op Fp8Rows,
//! cols % 16 == 0, block_cols a power of two of at least 16, weight and input 16-byte aligned, scales
//! 4-byte aligned, one word an output; else keep `expert_group`.
use crate::expert_ops::{GroupArguments,GroupEntry,Op};

const BR:usize=64;const BT:usize=64;const KC:usize=64;const THREADS:usize=256;
const WP:usize=80;const XP:usize=KC*4+16;
const SA:usize=BR*WP;const XA:usize=SA+BR*16;const STAGE:usize=XA+BT*XP;const SHARED:usize=2*STAGE;

#[no_mangle]
pub extern "ptx-kernel" fn fp8_rows_tile(p:GroupArguments) {
    // SAFETY: The host checks every part (shape, raw owners, input span) and keeps all owners and the table
    // alive through stream completion, as for expert_group. Weights, scales and inputs are immutable. Each
    // (part, row, token) output word has exactly one writing thread; the output is private and initialized
    // earlier on the stream, and its index first + row * tokens + token is inside the checked units. The
    // kernel traps unless the block has 256 threads, every part is Fp8Rows with the grid's rows and cols,
    // cols is a multiple of 16, block_cols a power of two of at least 16, weight and input are 16-byte
    // aligned, scales 4-byte aligned, and one word is an output. Every copy is guarded by row < rows,
    // token < tokens and column < cols, so it reads inside the checked extents (a 16-byte weight copy is
    // 16 whole columns of one row, a 16-byte input copy four whole columns of one token, a scale copy one
    // word of the part's scale table). Shared reads and writes stay inside the block's two static stages.
    // All 256 threads reach every barrier; loop bounds depend only on the part.
    unsafe {
        use core::arch::asm;
        macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        macro_rules! pair {($bits:expr)=>{{
            let (halves,first,second):(u32,f32,f32);let bits=$bits as u16;
            asm!("{{ .reg .b16 lo,hi; cvt.rn.f16x2.e4m3x2 {halves}, {bits}; mov.b32 {{lo,hi}}, {halves}; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",halves=out(reg32)halves,first=out(reg32)first,second=out(reg32)second,bits=in(reg16)bits);
            let _=halves;[first,second]
        }};}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        if core::arch::nvptx::_block_dim_x()!=THREADS as u32 || p.count==0 || p.words!=1 {core::arch::asm!("trap;");}
        let part=core::arch::nvptx::_block_idx_z() as usize;if part>=p.count as usize {return;}
        let table=core::slice::from_raw_parts(p.table as *const GroupEntry,p.count as usize);
        let first_part=&table[0];let e=&table[part];
        let (rows,cols,m)=(e.shape.rows as usize,e.shape.cols as usize,e.shape.tokens as usize);
        let (block_rows,block_cols)=(e.shape.block_rows as usize,e.shape.block_cols as usize);
        if e.shape.op!=Op::Fp8Rows || cols%16!=0 || !block_cols.is_power_of_two() || block_cols<16 || block_rows==0
            || rows!=first_part.shape.rows as usize || cols!=first_part.shape.cols as usize
            || (e.weight|p.input)&15!=0 || e.scales&3!=0 || p.stride as usize!=cols || (p.stride as u64*4)&15!=0 {core::arch::asm!("trap;");}
        let (token_tile,row_tile)=(core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize);
        if row_tile*BR>=rows || token_tile*BT>=m {return;}
        let smem=crate::ptx::shared::<SHARED>();
        let weights=e.weight as *const u8;let scales=e.scales as *const f32;
        let input=(p.input as *const f32).add(e.first_row as usize*cols);
        let shift=block_cols.trailing_zeros();let blocks=cols.div_ceil(block_cols);
        let chunks=cols.div_ceil(KC);
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE) as u32;
            // Weights: 64 bytes of each of 64 rows (four 16-byte copies per row).
            let mut item=tid;
            while item<BR*4 {let (r,q)=(item/4,item%4);let row=row_tile*BR+r;let col=c*KC+q*16;
                if row<rows && col<cols {crate::ptx::copy16(stage+(r*WP+q*16) as u32,weights.add(row*cols+col).cast());}
                item+=THREADS;}
            // Scales: the four 16-column groups of this step, one word per row and group.
            let mut item=tid;
            while item<BR*4 {let (r,g)=(item/4,item%4);let row=row_tile*BR+r;let col=c*KC+g*16;
                if row<rows && col<cols {crate::ptx::copy4(stage+(SA+r*16+g*4) as u32,scales.add(row/block_rows*blocks+(col>>shift)).cast());}
                item+=THREADS;}
            // Activations: 64 columns of each of 64 tokens (sixteen 16-byte copies per token).
            let mut item=tid;
            while item<BT*16 {let (t,q)=(item/16,item%16);let token=token_tile*BT+t;let col=c*KC+q*4;
                if token<m && col<cols {crate::ptx::copy16(stage+(XA+t*XP+q*16) as u32,input.add(token*cols+col).cast());}
                item+=THREADS;}
        };
        let (tx,ty)=(tid%16,tid/16);                                     // rows ty + 16 i; tokens tx + 16 j
        let mut acc=[[0.0f32;4];4];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE) as u32;
            let groups=((cols-c*KC).min(KC))/16;
            let mut g=0;
            while g<groups {
                let mut scale=[0.0f32;4];let mut bytes=[[0u32;4];4];
                rep4!(i,{
                    let r=ty+16*i;
                    scale[i]=crate::ptx::load(stage+(SA+r*16+g*4) as u32);
                    bytes[i]=crate::ptx::load4(stage+(r*WP+g*16) as u32);
                });
                rep4!(q,{
                    let mut w=[[0.0f32;4];4];
                    rep4!(i,{
                        let low=pair!(bytes[i][q]);let high=pair!(bytes[i][q]>>16);
                        w[i]=[mul!(low[0],scale[i]),mul!(low[1],scale[i]),mul!(high[0],scale[i]),mul!(high[1],scale[i])];
                    });
                    rep4!(j,{
                        let x=crate::ptx::load4(stage+(XA+(tx+16*j)*XP+g*64+q*16) as u32);
                        rep4!(el,{
                            let value=f32::from_bits(x[el]);
                            rep4!(i,{let product=mul!(w[i][el],value);add!(acc[i][j],product);});
                        });
                    });
                });
                g+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        rep4!(i,{
            let row=row_tile*BR+ty+16*i;
            if row<rows {
                rep4!(j,{
                    let token=token_tile*BT+tx+16*j;
                    if token<m {*(p.output as *mut f32).add(e.first as usize+row*m+token)=crate::expert_ops::round(acc[i][j]);}
                });
            }
        });
    }
}
