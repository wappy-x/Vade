//! Decode attention over expanded rows read in place through two pointer tables (keys, values). Same arithmetic
//! and order as `Op::ScoresIndirect` and `Op::ValuesIndirect`:
//!   score[head][slot] = bf16( bf16( sum_f q[head][f] * key[slot][head][f] ) * (1/sqrt(dim)) )   f ascending from +0.0
//!   out[head][f]      = bf16( sum_slot p[head][slot] * value[slot][head][f] )                    slot ascending from +0.0
//! each product one mul.rn.f32 and each step one add.rn.f32 (no FMA). What changes is the schedule:
//!   scores: lane = slot, 128 threads a block, 16-byte loads of the key row and of the query (the query is the same
//!           address in all lanes), one table entry a lane, and an L2 hint for the next 128-byte line of the key row;
//!           the chain per (head, slot) stays serial and ascending;
//!   values: one warp = (head, 128 features), each lane owns four consecutive features (16-byte loads, four independent
//!           chains), the slot chain of each feature stays serial and ascending.
//! Table entry (16 bytes, 4 words): u64 address of word (head 0, feature 0) of that slot's plane row, u32 head stride in
//! f32 words, u32 zero. Element (head, f) is at address + 4*(head*stride + f). The host validates every entry (live
//! same-context owner, span inside it, address % 16 == 0, stride % 4 == 0); these bodies trust the table.
//!
//! Launch: the same `Arguments` as `causal_operation`. ScoresTable: heads * ceil(rows/128) blocks of 128 threads.
//! ValuesTable: heads * dim/128 blocks of 32 threads. No shared memory. `causal_operation` selects the body by the op.
use crate::causal_ops::{Arguments,Op};

pub(crate) const SCORES_THREADS:u32=128;
pub(crate) const VALUES_THREADS:u32=32;

#[inline(always)]
pub(crate) fn scores(p:Arguments) {
    // SAFETY: The host validated the query (heads*dim words, 16-byte aligned), the table (live entries of 4 words) and
    // every row it names (see the contract above), keeps all owners live through completion, and gives the output
    // heads*rows private words. The body traps unless the block has 128 threads and the operation is ScoresTable.
    // Thread (block, lane) owns exactly one output word, head*rows+slot, and writes it once; slots past live write
    // 0.0 and read nothing. Key reads stay inside the validated span (dim words from head*stride); the L2 prefetch
    // names an address inside that span only (guarded by at+PF<dim).
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        const PF:usize=crate::causal_ops::ATTENTION_TABLE_SCORES_PREFETCH as usize;
        let shape=p.shape;let (heads,dim,rows,live)=(shape.heads as usize,shape.dim as usize,shape.rows as usize,shape.taps as usize);
        if _block_dim_x()!=SCORES_THREADS || shape.op!=Op::ScoresTable || dim%4!=0 || live>rows {asm!("trap;");}
        let tiles=rows.div_ceil(SCORES_THREADS as usize);let block=_block_idx_x() as usize;
        let head=block/tiles;let slot=block%tiles*SCORES_THREADS as usize+_thread_idx_x() as usize;
        if head>=heads || slot>=rows {return;}
        let output=(p.output as *mut f32).add(head*rows+slot);
        if slot>=live {output.write(0.0);return;}
        let entry=(p.input[1] as *const u32).add(slot*4);
        let address=u64::from(*entry)|u64::from(*entry.add(1))<<32;let stride=*entry.add(2) as usize;
        let key=(address as *const f32).add(head*stride);let query=(p.input[0] as *const f32).add(head*dim);
        let mut sum=0.0f32;let mut at=0;
        while at<dim {
            if PF!=0 && at%32==0 && at+PF<dim {asm!("prefetch.global.L2 [{p}];",p=in(reg64)key.add(at+PF),options(nostack,readonly));}
            let (k0,k1,k2,k3):(f32,f32,f32,f32);let (q0,q1,q2,q3):(f32,f32,f32,f32);
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)k0,b=out(reg32)k1,c=out(reg32)k2,d=out(reg32)k3,pointer=in(reg64)key.add(at),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q0,b=out(reg32)q1,c=out(reg32)q2,d=out(reg32)q3,pointer=in(reg64)query.add(at),options(nostack,readonly));
            add!(sum,mul!(q0,k0));add!(sum,mul!(q1,k1));add!(sum,mul!(q2,k2));add!(sum,mul!(q3,k3));at+=4;
        }
        let sum=f32::from_bits(crate::next_math::bf16(sum)<<16);
        output.write(f32::from_bits(crate::next_math::bf16(sum*(1.0/crate::qsa_common::root(dim as f32)))<<16));
    }
}

#[inline(always)]
pub(crate) fn values(p:Arguments) {
    // SAFETY: As for scores: the host validated the probabilities (heads*rows words), the table and every row it
    // names, and gives the output heads*dim private words. The body traps unless the block has 32 threads, the
    // operation is ValuesTable and dim is a multiple of 128. Thread (block, lane) owns the four output words
    // head*dim+tile*128+lane*4.., each written once. Reads: entries and probabilities below live (live <= rows), value
    // words inside the validated span (dim words from head*stride).
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        const F:usize=crate::causal_ops::ATTENTION_TABLE_VALUES_FEATURES as usize;
        let shape=p.shape;let (heads,dim,rows,live)=(shape.heads as usize,shape.dim as usize,shape.rows as usize,shape.taps as usize);
        if _block_dim_x()!=VALUES_THREADS || shape.op!=Op::ValuesTable || dim%(32*F)!=0 || live>rows {asm!("trap;");}
        let tiles=dim/(32*F);let block=_block_idx_x() as usize;let head=block/tiles;let first=block%tiles*32*F+_thread_idx_x() as usize*F;
        if head>=heads {return;}
        let table=p.input[1] as *const u32;let probabilities=(p.input[0] as *const f32).add(head*rows);
        let mut sum=[0.0f32;F];let mut slot=0;
        while slot<live {
            let entry=table.add(slot*4);let address=u64::from(*entry)|u64::from(*entry.add(1))<<32;let stride=*entry.add(2) as usize;
            let source=(address as *const f32).add(head*stride+first);let weight=*probabilities.add(slot);
            let mut e=0;while e<F {add!(sum[e],mul!(weight,*source.add(e)));e+=1;}
            slot+=1;
        }
        let output=(p.output as *mut f32).add(head*dim+first);
        let mut e=0;while e<F {output.add(e).write(f32::from_bits(crate::next_math::bf16(sum[e])<<16));e+=1;}
    }
}
