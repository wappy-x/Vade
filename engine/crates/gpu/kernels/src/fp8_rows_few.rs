//! Block-FP8 expert rows for 3 to 4 tokens a part (a draft verify pass): the same arithmetic and order as `fp8_rows_multi` and `fp8_rows::row!` (the
//! `expert_group` Fp8Rows path): every output `[row][token]` is ONE ascending FP32 chain over the columns from +0.0, each product `(e4m3(w) * scale) * x` made by two
//! separate `mul.rn.f32`, each step one separate `add.rn.f32`, no fused multiply-add, the same final BF16 round.
//!
//! `fp8_rows_multi` has a warp of 32 rows decode each weight once and keep up to 4 token chains; at 1,024 to 2,048 rows that is only 32 to 64 blocks of one computing warp.
//! Here a block takes 16 rows (a thread owns a row; lanes 16 to 31 of each warp idle) and warp `g` takes token `g` alone, so the 4 warps compute the tokens at the same
//! time (each decodes the weight itself) and a 1,024-row part runs on 64 blocks of up to 4 computing warps. Columns pass through shared memory in chunks of 128 through a
//! ring of 4 stages of `cp.async` copies with an L2 hint 4 chunks ahead.
//!
//! Launch: the same `GroupArguments` and table as `expert_group`; 128 threads; grid `ceil(rows/16) * parts` blocks (part-major), static shared memory.
//! All parts share rows and cols (the host checks); a part has 1 to 4 tokens.
use crate::expert_ops::{GroupArguments,GroupEntry,Op};

#[no_mangle]
pub extern "ptx-kernel" fn fp8_rows_few(launch:GroupArguments) {
    // SAFETY: The host checks every part (shape, raw owners, input span), that all parts share rows and cols, and keeps all owners and the table alive through stream
    // completion, as for expert_group. Weights, scales and inputs are immutable; each (part, row, token) output word has exactly one writer (the thread of that row in
    // the warp of that token) and the output is private. The body traps unless the block has 128 threads, the launch has one output word a unit and the stride of its
    // cols, and the block's part is an Fp8Rows part with the launch's rows and cols (cols a multiple of 16, block_cols a power of two of at least 16, block_rows at least 1),
    // 1 to 4 tokens, a 16-byte aligned weight, a 4-byte aligned scale array and a 16-byte aligned input, and its output range lies inside `units`. Global reads stay inside
    // the row (weights: row * cols + column below cols with 16-byte pieces that start below cols, cols being a multiple of 16), the scale array (ceil(rows/block_rows) *
    // ceil(cols/block_cols) words) and the part's own input rows (a copy starts below cols and covers four whole columns). Every shared write is a cp.async into the ring
    // slot `chunk % D`, inside the block's static array. A slot is refilled only after the barrier that follows every thread's reads of its previous chunk, and is read only
    // after cp.async.wait_group and a barrier that follow its fill: every shared read follows a write. A thread reads only the weight row of its lane and the token row of
    // its warp, and those were written for this chunk (rows and tokens past the shape are neither copied nor read). All threads reach every barrier; the loop bounds depend
    // only on the part's shape. The output word is stored through a raw pointer; no reference is built over the output.
    unsafe {
        use core::arch::asm;
        // Ring depth, rows a block, columns a chunk, L2 hint distance in chunks, tokens at most, bytes of a padded weight row.
        const D:usize=4;const RB:usize=16;const KC:usize=128;const PF:usize=4;const TM:usize=4;const PITCH:usize=KC+16;const PIECES:usize=KC/16;
        const WEIGHT_BYTES:usize=RB*PITCH;const STAGE:usize=WEIGHT_BYTES+TM*KC*4;
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        macro_rules! pair {($bits:expr)=>{{
            let (halves,first,second):(u32,f32,f32);let bits=$bits as u16;
            asm!("{{ .reg .b16 lo,hi; cvt.rn.f16x2.e4m3x2 {halves}, {bits}; mov.b32 {{lo,hi}}, {halves}; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",halves=out(reg32)halves,first=out(reg32)first,second=out(reg32)second,bits=in(reg16)bits,options(nomem,nostack));
            let _=halves;[first,second]
        }};}
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        if launch.count==0 || core::arch::nvptx::_block_dim_x()!=128 {asm!("trap;");}
        let table=core::slice::from_raw_parts(launch.table as *const GroupEntry,launch.count as usize);
        let rows_launch=table[0].shape.rows as usize;let cols_launch=table[0].shape.cols as usize;
        if rows_launch==0 || cols_launch==0 || cols_launch%16!=0 || launch.words!=1 || launch.stride as usize!=cols_launch || launch.input&15!=0 {asm!("trap;");}
        let per_part=rows_launch.div_ceil(RB);let block=core::arch::nvptx::_block_idx_x() as usize;
        let (part,row0)=(block/per_part,block%per_part*RB);
        if part>=table.len() {return;}
        let e=table[part];let shape=e.shape;let rows=shape.rows as usize;let k=shape.cols as usize;let m=shape.tokens as usize;
        let block_cols=shape.block_cols as usize;let block_rows=shape.block_rows as usize;
        if shape.op!=Op::Fp8Rows || rows!=rows_launch || k!=cols_launch || m==0 || m>TM || block_cols<16 || !block_cols.is_power_of_two() || block_rows==0
            || e.weight&15!=0 || e.scales&3!=0 || e.first as usize+rows*m>launch.units as usize {asm!("trap;");}
        let shift=block_cols.trailing_zeros();let blocks=k.div_ceil(block_cols);
        let smem=crate::ptx::shared::<{D*STAGE}>();
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (lane,token)=(tid%32,tid/32);
        // Lanes past the block's rows and a warp past the last token only copy: they reach every barrier and compute nothing.
        let active=lane<RB && row0+lane<rows && token<m;
        let weights=e.weight as *const u8;let scales=e.scales as *const f32;
        let input=(launch.input as *const f32).add(e.first_row as usize*k);
        let chunks=k.div_ceil(KC);
        let scale_row=if active {scales.add((row0+lane)/block_rows*blocks)} else {scales};
        // Copy chunk `c` of this block's weight rows and of every token row into slot `c % D`.
        let issue=|c:usize| {
            let stage=smem+((c%D)*STAGE) as u32;let col0=c*KC;
            let mut item=tid;
            while item<RB*PIECES {
                let (r,piece)=(item/PIECES,item%PIECES);let col=col0+piece*16;
                if col<k && row0+r<rows {crate::ptx::copy16(stage+(r*PITCH+piece*16) as u32,weights.add((row0+r)*k+col).cast());}
                item+=128;
            }
            item=tid;
            while item<m*(KC/4) {
                let (t,q)=(item/(KC/4),item%(KC/4));let col=col0+q*4;
                if col<k {crate::ptx::copy16(stage+(WEIGHT_BYTES+t*KC*4+q*16) as u32,input.add(t*k+col).cast());}
                item+=128;
            }
        };
        // Hint the cache line of chunk `c` of every weight row into L2. Reads nothing, writes nothing.
        let hint=|c:usize| {
            let col0=c*KC;
            if col0<k {
                let mut r=tid;
                while r<RB {
                    if row0+r<rows {asm!("prefetch.global.L2 [{p}];",p=in(reg64)weights.add((row0+r)*k+col0),options(nostack,readonly));}
                    r+=128;
                }
            }
        };
        let mut sum=0.0f32;
        let mut c=0;
        while c<D-1 {if c<chunks {issue(c);}crate::ptx::commit();c+=1;}
        c=D-1;
        while c<PF {hint(c);c+=1;}
        c=0;
        while c<chunks {
            pending!(D-2);crate::ptx::sync();
            if c+D-1<chunks {issue(c+D-1);}crate::ptx::commit();
            hint(c+PF);
            if active {
                let stage=smem+((c%D)*STAGE) as u32;let col0=c*KC;let pieces=(k-col0).min(KC)/16;
                // Constant indices only (unrolled by the macros below), so the scales and the weight words stay in registers.
                macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
                macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{rep4!(w,{let $v:usize=4+w;$body});}}};}
                // This chunk's scales, all loaded before the pieces are computed.
                let mut scale=[0.0f32;PIECES];
                rep8!(i,{if i<pieces {scale[i]=*scale_row.add((col0+i*16)>>shift);}});
                rep8!(i,{
                    if i<pieces {
                        let bytes=crate::ptx::load4(stage+(lane*PITCH+i*16) as u32);
                        rep4!(word,{
                            // Four columns: decode and scale, then multiply by this warp's token and add, in column order.
                            let (low,high)=(pair!(bytes[word]),pair!(bytes[word]>>16));
                            let ws=[mul!(low[0],scale[i]),mul!(low[1],scale[i]),mul!(high[0],scale[i]),mul!(high[1],scale[i])];
                            let x=crate::ptx::load4(stage+(WEIGHT_BYTES+token*KC*4+(i*16+word*4)*4) as u32);
                            add!(sum,mul!(ws[0],f32::from_bits(x[0])));add!(sum,mul!(ws[1],f32::from_bits(x[1])));
                            add!(sum,mul!(ws[2],f32::from_bits(x[2])));add!(sum,mul!(ws[3],f32::from_bits(x[3])));
                        });
                    }
                });
            }
            c+=1;
        }
        if active {*(launch.output as *mut f32).wrapping_add(e.first as usize+(row0+lane)*m+token)=crate::small_math::round(sum);}
    }
}
