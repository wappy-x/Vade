//! The pair split's two small kernels, in their own module so the existing kernels keep their code.
//! `nv_quarter_fold`: the expert fold inside each column quarter and the join of an owner's quarters (QUARTER-FOLD
//! v1; CPU twin `tc_twin_fold`). `tp_reduce_take`: the REDUCE of two owners, one exchange a sublayer.
//! Every product is `mul.rn.f32` and every sum is `add.rn.f32`, written as instructions: nothing here can be fused
//! into a multiply-add or reordered.
use core::arch::asm;
use core::arch::nvptx::{_block_dim_x,_block_idx_x,_grid_dim_x,_thread_idx_x};
use crate::pair_split_abi::{QuarterFold,FOLD_MAX_EXPERTS,FOLD_PLAIN,FOLD_QUARTER_MAJOR,FOLD_ROUND,FOLD_SCALES_BY_ID,REDUCE_LOWER,REDUCE_ROUND,REDUCE_SCALE};

macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
macro_rules! add {($a:expr,$b:expr)=>{{let value:f32;asm!("add.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}

/// One token row. For each output row and each of the owner's quarters q: Q[q] = +0.0; for each routed expert in
/// fold order Q[q] = fl(Q[q] + fl(fl(a[e][q] * k[e]) * w[e])); then Q[q] = fl(Q[q] + s[q]) when the layer has a
/// shared expert. The output is the join of the owner's quarters, neighbors first: Q[0]; fl(Q[0] + Q[1]);
/// fl(fl(Q[0] + Q[1]) + fl(Q[2] + Q[3])). With FOLD_ROUND it is rounded to BF16.
/// Fold order = the rule of today's fold (`latent_ops`, `Op::ExpertFold`): each visit takes the smallest id above
/// the last visited id and the first slot that holds it; with no id left above, slot 0.
/// With FOLD_PLAIN there is no fold: Q[q] is the raw quarter accumulator `acc[row][q]` itself.
/// With FOLD_QUARTER_MAJOR the shared operand (and `acc` of FOLD_PLAIN) is read at `[q][row]`: the same words in the
/// layout of one job a quarter (block FP8, MXFP8). The arithmetic and its order do not change.
/// With FOLD_SCALES_BY_ID the scale of a slot is `scales[id of the slot]`: `scales` then holds every expert of the layer.
#[no_mangle]
pub extern "ptx-kernel" fn nv_quarter_fold(p:QuarterFold) {resident_nv_quarter_fold(p)}
/// The body of `nv_quarter_fold`, for its own entry and for its recorded form (`resident_launch_quarter_fold`).
#[inline(always)]
pub(crate) fn resident_nv_quarter_fold(p:QuarterFold) {
    // SAFETY: Backend::nv_quarter_fold checks the launch. `ids`, `weights` and `scales` each cover `experts` FP32
    // words (1 to FOLD_MAX_EXPERTS); with FOLD_SCALES_BY_ID `scales` covers `pool` words instead (1 to the route's
    // MAX_EXPERTS), and an id indexes it only after `selected` found it a whole number below `pool` (else a trap); `acc` covers experts * rows * quarters words (rows * quarters with FOLD_PLAIN,
    // which reads no other input); `shared` is null or covers
    // rows * quarters words (both layouts of FOLD_QUARTER_MAJOR index below rows * quarters: q < quarters and
    // row < rows); `output` covers `rows` words and overlaps none of the inputs; `quarters` is 1, 2 or 4.
    // All owners live through stream completion and the inputs are not written while this launch runs. The grid has
    // ceil(rows / block size) blocks, so thread t of block b owns output row b * size + t and is its only writer
    // (rows past the end return). The shared scratch holds FOLD_MAX_EXPERTS words and is this kernel's only shared
    // allocation: thread 0 of each block writes the visiting order there before the barrier, and every thread reads
    // it only after the barrier, which every thread of the block executes once. A slot read from it is below
    // `experts`. The body traps on any other launch.
    unsafe {
        let (tid,threads)=(_thread_idx_x() as usize,_block_dim_x() as usize);
        let (experts,rows,quarters)=(p.experts as usize,p.rows as usize,p.quarters as usize);
        if experts==0 || experts>FOLD_MAX_EXPERTS as usize || rows==0 || !(quarters==1 || quarters==2 || quarters==4) {asm!("trap;");}
        let order=crate::ptx::shared::<{4*FOLD_MAX_EXPERTS as usize}>();
        let plain=p.flags&FOLD_PLAIN!=0;let major=p.flags&FOLD_QUARTER_MAJOR!=0;let by_id=p.flags&FOLD_SCALES_BY_ID!=0;
        if by_id && (plain || p.pool==0 || p.pool>crate::device_route_abi::MAX_EXPERTS) {asm!("trap;");}
        if tid==0 && !plain {
            let ids=p.ids as *const f32;
            let (mut previous,mut started,mut visit)=(0.0f32,false,0usize);
            while visit<experts {
                let (mut selected,mut best,mut slot)=(0usize,f32::INFINITY,0usize);
                while slot<experts {
                    let id=*ids.add(slot);
                    if (!started || id>previous) && id<best {selected=slot;best=id;}
                    slot+=1;
                }
                crate::ptx::store(order+4*visit as u32,f32::from_bits(selected as u32));
                previous=best;started=true;visit+=1;
            }
        }
        crate::ptx::sync();
        let row=_block_idx_x() as usize*threads+tid;
        if row>=rows {return;}
        let (weights,scales,acc,shared)=(p.weights as *const f32,p.scales as *const f32,p.acc as *const f32,p.shared as *const f32);
        let (mut q0,mut q1,mut q2,mut q3)=(0.0f32,0.0f32,0.0f32,0.0f32);
        let mut q=0usize;
        while q<quarters {
            let (mut sum,mut visit)=(0.0f32,if plain {experts} else {0usize});
            let own=if major {q*rows+row} else {row*quarters+q};
            if plain {sum=*acc.add(own);}
            while visit<experts {
                let slot=crate::ptx::load(order+4*visit as u32).to_bits() as usize;
                let scale=if by_id {
                    let Some(expert)=crate::device_route_abi::selected(*(p.ids as *const f32).add(slot),p.pool) else {core::arch::nvptx::trap();};
                    *scales.add(expert as usize)
                } else {*scales.add(slot)};
                let scaled=mul!(*acc.add((slot*rows+row)*quarters+q),scale);
                let term=mul!(scaled,*weights.add(slot));
                sum=add!(sum,term);
                visit+=1;
            }
            if !plain && !shared.is_null() {sum=add!(sum,*shared.add(own));}
            match q {0=>q0=sum,1=>q1=sum,2=>q2=sum,_=>q3=sum}
            q+=1;
        }
        let joined=match quarters {1=>q0,2=>add!(q0,q1),_=>{let (low,high)=(add!(q0,q1),add!(q2,q3));add!(low,high)}};
        *(p.output as *mut f32).add(row)=if p.flags&FOLD_ROUND!=0 {crate::bf16::rounded(joined)} else {joined};
    }
}

/// The REDUCE of a pair. Every block's thread 0 waits for the peer's message as `tp_mailbox_take` does: until
/// `seq_word` (the receive slot's trailer word) holds the low word of `seq`, at most the low 47 bits of `limit_ns`
/// (its high 16 bits cap the poll backoff, 0 = 2048 ns), or until `error` is set; on a timeout it sets `error` and
/// the block writes nothing. Then for each of `words` words: sum = fl(lower + higher), where the lower owner's
/// partial is the first operand on both owners (`own` with REDUCE_LOWER, else `peer`); with REDUCE_SCALE
/// sum = fl(sum * scale) (the scale is the high 32 bits of `mode`); with REDUCE_ROUND the result is rounded to BF16.
/// Both owners run the same operations on the same bits, so both write the same words.
#[no_mangle]
pub extern "ptx-kernel" fn tp_reduce_take(out:*mut f32,own:*const f32,words:u64,peer:*const f32,seq_word:*const u32,seq:u64,error:*mut u32,limit_ns:u64,mode:u64) {
    resident_tp_reduce_take(out,own,words,peer,seq_word,seq,error,limit_ns,mode)
}
/// The body of `tp_reduce_take`, for its own entry and for its recorded form (`resident_launch_reduce_take`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn resident_tp_reduce_take(out:*mut f32,own:*const f32,words:u64,peer:*const f32,seq_word:*const u32,seq:u64,error:*mut u32,limit_ns:u64,mode:u64) {
    // SAFETY: Backend::tp_reduce_take checks the launch. `own` and `out` each cover `words` FP32 words of retained
    // device vectors; they are one vector or do not overlap. `peer` is the receive slot's `words` words that end at
    // its trailer (mapped host memory), and `seq_word` is the trailer's last word; `error` is the mailbox's aligned
    // header word. All live through ordered completion. The peer's rows are read only after the acquire of the
    // sequence word, which the wire writes after every row of the message (tp_mailbox_host::inline_span). `words` is
    // a multiple of 4 and `out`, `own` and `peer` are 16-byte aligned (checked by the host; the body traps if not).
    // A group of four words has one thread (the step is grid times block size): it reads its four words of `own`
    // and of `peer` before it writes its four words of `out`, so `out` may be `own`. The shared status word has one writer a block before the barrier. A block that timed
    // out or saw the error reads and writes no row.
    unsafe {
        let sh=crate::ptx::shared::<4>();
        if _thread_idx_x()==0 {
            let start:u64;asm!("mov.u64 {t}, %globaltimer;",t=out(reg64)start);
            let (limit,cap)=(limit_ns&((1<<47)-1),(limit_ns>>48) as u32);
            let cap=if cap==0 {2048} else {cap};
            let (mut ok,mut pause)=(1u32,32u32.min(cap));
            loop {
                let seen:u32;asm!("ld.acquire.sys.global.u32 {v}, [{p}];",v=out(reg32)seen,p=in(reg64)seq_word);
                if seen==seq as u32 {break;}
                let failed:u32;asm!("ld.acquire.sys.global.u32 {v}, [{p}];",v=out(reg32)failed,p=in(reg64)error);
                let now:u64;asm!("mov.u64 {t}, %globaltimer;",t=out(reg64)now);
                if failed!=0 || now.wrapping_sub(start)>limit {
                    asm!("st.release.sys.global.u32 [{p}], {v};",p=in(reg64)error,v=in(reg32)1u32);
                    ok=0;break;
                }
                asm!("nanosleep.u32 {t};",t=in(reg32)pause);
                pause=(pause*2).min(cap);
            }
            crate::ptx::store(sh,f32::from_bits(ok));
        }
        crate::ptx::sync();
        if crate::ptx::load(sh).to_bits()==0 {return;}
        let words=words as usize;
        // Four words a thread and a step (one 16-byte load of each partial sum, one 16-byte store), as the plain take.
        if words%4!=0 || (out as usize|own as usize|peer as usize)%16!=0 {asm!("trap;");}
        let (lower,scaled,round,scale)=(mode&REDUCE_LOWER!=0,mode&REDUCE_SCALE!=0,mode&REDUCE_ROUND!=0,f32::from_bits((mode>>32) as u32));
        macro_rules! word {($mine:expr,$theirs:expr)=>{{
            let mut sum=if lower {add!($mine,$theirs)} else {add!($theirs,$mine)};
            if scaled {sum=mul!(sum,scale);}
            if round {crate::bf16::rounded(sum)} else {sum}
        }};}
        let step=(_grid_dim_x()*_block_dim_x()) as usize;
        let mut i=(_block_idx_x()*_block_dim_x()+_thread_idx_x()) as usize;
        while i<words/4 {
            let (a0,a1,a2,a3,b0,b1,b2,b3):(f32,f32,f32,f32,f32,f32,f32,f32);
            asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a0,b=out(reg32)a1,c=out(reg32)a2,d=out(reg32)a3,p=in(reg64)own.add(4*i));
            asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)b0,b=out(reg32)b1,c=out(reg32)b2,d=out(reg32)b3,p=in(reg64)peer.add(4*i));
            let (r0,r1,r2,r3)=(word!(a0,b0),word!(a1,b1),word!(a2,b2),word!(a3,b3));
            asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",p=in(reg64)out.add(4*i),a=in(reg32)r0,b=in(reg32)r1,c=in(reg32)r2,d=in(reg32)r3);
            i+=step;
        }
    }
}
