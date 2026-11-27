//! Block-FP8 expert rows for 2 to 8 tokens a part (a draft verify pass, several chats in one step): ONE read and ONE decode of a weight feeds the
//! chains of all tokens. Same arithmetic and order as `fp8_rows::row!` (the `expert_group` Fp8Rows path) and as `fp8_rows_small`, `fp8_rows_few` and
//! `fp8_rows_multi`: every output `[row][token]` is ONE ascending FP32 chain over the columns from +0.0, each product `(e4m3(w) * scale) * x` made by
//! two separate `mul.rn.f32` (the activation words are used as they are), each step one separate `add.rn.f32`, no fused multiply-add, and the same
//! final BF16 round.
//!
//! What changes is who does the work. `fp8_rows_small` runs the one-row tile once for each token, so every token copies and decodes the weights
//! again. `fp8_rows_few` and `fp8_rows_multi` copy a weight once, but each token's warp decodes and scales it again in the thread that also runs
//! the chain, so a launch costs the instructions of one warp (gate and up 1,024 x 4,096, one Spark: 50 us at 2 to 4 tokens against 32 us for the
//! one-row tile). Here a block owns R=16 weight rows of one part, as the one-row tile does:
//! - R*QN producer threads copy the weights with asynchronous copies (D stages deep, an L2 hint PF chunks ahead), decode them and make the first
//!   product `ws = e4m3(w) * scale` ONCE into shared memory. That value does not depend on the token.
//!   They also copy the chunk's activations of every token into shared memory (a chain lane that loaded them from global memory stalled on
//!   every load: 110 us for kv_a 256 x 4,096 against 18 us for the one-row tile, one Spark, run `shareb-fp8-time-01`). For 2 to 4 tokens the
//!   activations ride in the weights' copy group, D stages deep. For 5 to 8 tokens they are copied one chunk later, in a group of their own,
//!   into 2 stages: the activations come from the cache, not from memory, and the smaller region keeps the block's shared memory at that of
//!   4 tokens (5 and 6 tokens) or 2 KB above it (7 and 8).
//! - A chain warp holds 2 tokens: lane `l` owns weight row `l % 16` and token `2 * warp + l / 16`. It loads `ws` and its token's activation,
//!   makes the second product `ws * x` and adds it to its own chain, in column order. A part of `m` tokens has `ceil(m / 2)` chain warps.
//!
//! So the memory traffic and the decode of a launch are those of ONE row, for any token count of 2 to 8, and a chain is alone in its lane.
//!
//! Launch: the same `GroupArguments` and table as `expert_group`; `32 * ceil(m / 2) + 128` threads, where `m` is the token count of every part;
//! grid `ceil(rows/16) * parts` blocks (part-major, as `fp8_rows_multi`); dynamic shared memory of `shared_bytes(m)` bytes. All parts share
//! rows, cols and the token count (the host checks).
use crate::expert_ops::{GroupArguments,GroupEntry,Op};

/// Most tokens a part may have.
pub const MAX_TOKENS:usize=8;
// Rows a block, 16-column pieces a row and chunk, ring depth of the copies, chunks the L2 hint runs ahead.
const R:usize=16;const QN:usize=8;const D:usize=3;const PF:usize=2;
// Columns a chunk, producer threads, words of a product row and of an activation row (4 words of padding: the lanes of a vector load use every bank twice).
const CK:usize=QN*16;const ITEMS:usize=R*QN;const PW:usize=CK+4;
// Shared layout in bytes: raw weight stages, scale stages, two product buffers, then the activation stages (a slot for each 2 tokens of a chain warp;
// `x_stages` of them).
const RAW_AT:usize=0;const SCALE_AT:usize=RAW_AT+D*R*CK;const PROD_AT:usize=SCALE_AT+D*ITEMS*4;const X_AT:usize=PROD_AT+2*R*PW*4;

#[export_name="fp8_rows_share"]
pub extern "ptx-kernel" fn entry_fp8_rows_share(launch:GroupArguments) {fp8_rows_share(launch)}
#[inline(always)]
pub(crate) fn fp8_rows_share(launch:GroupArguments) {
    // SAFETY: The host checks every part (shape, raw owners, input span), that all parts share rows, cols and the token count, and keeps all
    // owners and the table alive through stream completion, as for expert_group. Weights, scales and inputs are immutable; each (part, row, token)
    // output word has exactly one writer (the chain lane of that row and token in the block that owns the row) and the output is private and
    // initialized earlier on the stream. The body traps unless the launch has one output word a unit and the stride of its cols, and the block's
    // part is an Fp8Rows part with the launch's rows and cols (cols a multiple of 16, block_cols a power of two of at least 16, block_rows at
    // least 1), 2 to 8 tokens, a block of 32 * ceil(tokens / 2) + 128 threads, a dynamic shared region that holds the layout below, a 16-byte
    // aligned weight, a 4-byte aligned scale array and a 16-byte aligned input, and its output range lies inside `units`. Global reads stay below
    // rows*cols weight bytes, ceil(rows/block_rows)*ceil(cols/block_cols) scale words and the part's own input rows: a producer copies a weight
    // piece only when its row is below rows and its 16 columns start below cols (cols is a multiple of 16), and an activation piece of 4 words
    // only of a token below the part's count at a column below cols (a multiple of 4). The L2 hint reads nothing and names an address inside the
    // row. Shared addresses stay inside the dynamic region (the only shared allocation of this kernel, so it starts at 0): raw weight stages
    // (D * R rows of CK bytes), scale stages (D * R * QN words), two product buffers (R rows of CK + 4 words), activation stages (`x_stages`
    // stages of 2 * ceil(tokens / 2) rows of CK + 4 words). Every shared read follows a write: a producer reads only the raw piece and the scale word it
    // copied itself for that chunk, after its own wait; a chain lane reads product words only of a row below rows and activation words only of a
    // token below the count, of whole 16-column groups below cols, which the producers stored or copied for that chunk and waited for before
    // the barrier in between. The two product buffers alternate by chunk and a copy stage is refilled D chunks later (an activation stage of 5
    // to 8 tokens 2 chunks later, by a copy issued after the barrier that ends the chunk that read it), so a buffer is written only after the
    // barrier that follows its last read. All threads of a block reach every barrier the same number of times (the loop bound
    // depends only on cols); a block whose part does not exist returns before any barrier, all threads alike.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        macro_rules! pair {($bits:expr)=>{{
            let (halves,first,second):(u32,f32,f32);let bits=$bits as u16;
            asm!("{{ .reg .b16 lo,hi; cvt.rn.f16x2.e4m3x2 {halves}, {bits}; mov.b32 {{lo,hi}}, {halves}; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",halves=out(reg32)halves,first=out(reg32)first,second=out(reg32)second,bits=in(reg16)bits,options(nomem,nostack));
            let _=halves;[first,second]
        }};}
        macro_rules! store4 {($address:expr,$a:expr,$b:expr,$c:expr,$d:expr)=>{asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",p=in(reg32)$address,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,d=in(reg32)$d)};}
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        if launch.count==0 {asm!("trap;");}
        let table=core::slice::from_raw_parts(launch.table as *const GroupEntry,launch.count as usize);
        let rows_launch=table[0].shape.rows as usize;let cols_launch=table[0].shape.cols as usize;
        if rows_launch==0 || cols_launch==0 || cols_launch%16!=0 || launch.words!=1 || launch.stride as usize!=cols_launch || launch.input&15!=0 {asm!("trap;");}
        let per_part=rows_launch.div_ceil(R);let block=_block_idx_x() as usize;
        let (part,row0)=(block/per_part,block%per_part*R);
        if part>=table.len() {return;}
        let e=table[part];let shape=e.shape;let rows=shape.rows as usize;let k=shape.cols as usize;let m=shape.tokens as usize;
        let block_cols=shape.block_cols as usize;let block_rows=shape.block_rows as usize;
        if shape.op!=Op::Fp8Rows || rows!=rows_launch || k!=cols_launch || m<2 || m>MAX_TOKENS || block_cols<16 || !block_cols.is_power_of_two() || block_rows==0
            || e.weight&15!=0 || e.scales&3!=0 || e.first as usize+rows*m>launch.units as usize {asm!("trap;");}
        // Chain warps first, then the producers. `held`: the token slots of the chain warps.
        let held=2*m.div_ceil(2);let chain=16*held;
        // 5 to 8 tokens: the activations go one chunk later into 2 stages (see the module text).
        let late=held>4;let x_stages=if late {2} else {D};
        // The activation stage of a chunk, with no division by a run-time value (a 64-bit remainder a chunk cost the small tables 2 us a launch).
        let x_stage=|c:usize| if late {c&1} else {c%D};
        if _block_dim_x() as usize!=chain+ITEMS || (dynamic as usize)<X_AT+x_stages*held*PW*4 {asm!("trap;");}
        let shift=block_cols.trailing_zeros();let blocks=k.div_ceil(block_cols);
        let chunks=k.div_ceil(CK);
        let tid=_thread_idx_x() as usize;
        let weights=e.weight as *const u8;let scales=e.scales as *const f32;
        let input=(launch.input as *const f32).add(e.first_row as usize*k);
        let producer=tid>=chain;let pt=tid.wrapping_sub(chain);
        let (item_row,item_q)=(pt/QN,pt%QN);
        let row=row0+item_row;
        // Producer side: the activations of chunk c of every token, 4 words a copy: 32 copies a token, spread over the producers.
        let copy_x=|c:usize| {
            if c<chunks {
                let col0=c*CK;let stage=x_stage(c);
                let mut item=pt;
                while item<held*(CK/4) {
                    let (t,piece)=(item/(CK/4),item%(CK/4));
                    if t<m && col0+piece*4<k {crate::ptx::copy16_cached((X_AT+stage*held*PW*4+t*PW*4+piece*16) as u32,input.add(t*k+col0+piece*4).cast());}
                    item+=ITEMS;
                }
            }
        };
        // Producer side: issue the copies of chunk c (always commits the same groups so the counts stay uniform): the weights and scales of
        // chunk c, with the activations of chunk c (2 to 4 tokens) or, in an earlier group of their own, those of chunk c - 1 (5 to 8 tokens).
        // After `issue(c)` a wait for all groups but the last leaves complete: the weights of chunk c - 1 and the activations of chunk c - 1.
        let issue=|c:usize| {
            if late {if c>=1 {copy_x(c-1);}crate::ptx::commit();}
            if c<chunks {
                let col0=c*CK;let stage=c%D;
                if row<rows && col0+item_q*16<k {
                    crate::ptx::copy16((RAW_AT+stage*R*CK+item_row*CK+item_q*16) as u32,weights.add(row*k+col0+item_q*16).cast());
                    // Pull this row's line PF chunks ahead into L2: one thread a row. Reads nothing, writes nothing.
                    if item_q==0 && col0+PF*CK<k {asm!("prefetch.global.L2 [{p}];",p=in(reg64)weights.add(row*k+col0+PF*CK),options(nostack,readonly));}
                    crate::ptx::copy4((SCALE_AT+stage*ITEMS*4+pt*4) as u32,scales.add(row/block_rows*blocks+((col0+item_q*16)>>shift)).cast());
                }
                if !late {copy_x(c);}
            }
            crate::ptx::commit();
        };
        // Producer side: the first products of chunk c, `ws[row][column]` in column order: decode and scale ONCE for every token.
        let produce=|c:usize,buffer:usize| {
            if c<chunks && row<rows && c*CK+item_q*16<k {
                let stage=c%D;
                let raw=crate::ptx::load4((RAW_AT+stage*R*CK+item_row*CK+item_q*16) as u32);
                let scale=crate::ptx::load((SCALE_AT+stage*ITEMS*4+pt*4) as u32);
                rep4!(slot,{
                    let bytes=raw[slot];
                    let (first,second)=(pair!(bytes),pair!(bytes>>16));
                    let (w0,w1,w2,w3)=(mul!(first[0],scale),mul!(first[1],scale),mul!(second[0],scale),mul!(second[1],scale));
                    store4!((PROD_AT+buffer*R*PW*4+item_row*PW*4+(item_q*16+slot*4)*4) as u32,w0.to_bits(),w1.to_bits(),w2.to_bits(),w3.to_bits());
                });
            }
        };
        // Chain side: lane `l` of chain warp `w` owns weight row `l % 16` and token `2 * w + l / 16`.
        let (lane_row,token)=(tid%R,2*(tid/32)+(tid%32)/R);
        let own=!producer && row0+lane_row<rows && token<m;
        let mut sum=0.0f32;
        // Prologue: chunks 0..D-2 in flight, chunk 0 arrived, its first products made.
        if producer {
            let mut c=0;while c<D-1 {issue(c);c+=1;}
            pending!(D-2);
            produce(0,0);
        }
        crate::ptx::sync();
        let mut c=0;
        while c<chunks {
            if producer {
                issue(c+D-1);
                pending!(D-2);
                produce(c+1,(c+1)&1);
            } else if own {
                let col0=c*CK;let groups=(k-col0).min(CK)/16;
                let wbase=(PROD_AT+(c&1)*R*PW*4+lane_row*PW*4) as u32;
                let xbase=(X_AT+x_stage(c)*held*PW*4+token*PW*4) as u32;
                // The second products of 16 columns, in column order.
                macro_rules! products {($q:expr)=>{{
                    let mut p=[0.0f32;16];
                    rep4!(slot,{
                        let at=(($q)*16+slot*4)*4;
                        let (ws,x)=(crate::ptx::load4(wbase+at as u32),crate::ptx::load4(xbase+at as u32));
                        rep4!(j,{p[slot*4+j]=mul!(f32::from_bits(ws[j]),f32::from_bits(x[j]));});
                    });
                    p
                }};}
                let mut q=0;
                while q<groups {
                    let p=products!(q);
                    rep4!(slot,{rep4!(j,{add!(sum,p[slot*4+j]);});});
                    q+=1;
                }
            }
            crate::ptx::sync();
            c+=1;
        }
        if own {*(launch.output as *mut f32).add(e.first as usize+(row0+lane_row)*m+token)=crate::small_math::round(sum);}
    }
}
/// Dynamic shared bytes of a launch whose parts have `tokens` tokens.
#[allow(dead_code)]
pub const fn shared_bytes(tokens:usize)->usize {X_AT+(if tokens>4 {2} else {D})*2*tokens.div_ceil(2)*PW*4}
