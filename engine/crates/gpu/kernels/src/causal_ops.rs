//! Checked single-token operations. Each thread owns one complete output unit.
use crate::causal_ops::{Arguments,unit};
// The scalar reference keeps its ascending scans. Independent scores are
// parallel, and the scans use shared storage instead of the output allocation.
macro_rules! shared_route {($p:expr,$words:expr)=>{{
    let p=$p;let s=p.shape;let (r,h,d,k)=(s.rows as usize,s.heads as usize,s.dim as usize,s.cols as usize);
    if core::arch::nvptx::_block_idx_x()!=0 {return;}
    let thread=core::arch::nvptx::_thread_idx_x() as usize;let threads=core::arch::nvptx::_block_dim_x() as usize;
    let base=crate::ptx::shared::<{crate::latent_ops::ROUTE_SHARED_BYTES}>();
    let choice=base+(r*4) as u32;let groups=choice+(r*4) as u32;let kept=groups+(h*4) as u32;
    let input=p.input[0] as *const f32;let weight=p.weight as *const f32;let output=p.output as *mut f32;
    // One group, one kept group, at most `rows` picks, room for the slot words: the picks are found by all threads.
    let ranked=crate::latent_ops::route_ranked(s);
    let slots=kept+(h*4) as u32;let flag=slots+(k*4) as u32;
    if ranked && thread==0 {crate::ptx::store(flag,0.0);}
    let mut expert=thread;let mut ordered=true;
    while expert<r {let score=crate::latent_ops::sigmoid(*input.add(expert));let chosen=score+*weight.add(expert);
        ordered&=chosen.to_bits()&0x7f80_0000!=0x7f80_0000;
        crate::ptx::store(base+(expert*4) as u32,score);
        crate::ptx::store(choice+(expert*4) as u32,chosen);expert+=threads;}
    let mut word=2*k+thread;while word<$words {output.add(word).write(0.0);word+=threads;}
    core::arch::nvptx::_syncthreads();
    if ranked {
        // A choice that is not finite (NaN or an infinity) keeps the scalar scans below, with their exact behavior.
        if !ordered {crate::ptx::store(flag,1.0);}
        core::arch::nvptx::_syncthreads();
        if crate::ptx::load(flag)==0.0 {
            // The scalar scans pick, slot by slot, the largest remaining choice, the lowest expert among equals
            // (`value>best` in ascending order). With every choice finite that is the order (choice descending,
            // expert ascending): slot s holds the expert that has exactly s experts before it in that order.
            // Each thread counts that for its own experts, in one pass with no branch. Comparisons only: no arithmetic moves.
            let mut expert=thread;
            while expert<r {let value=crate::ptx::load(choice+(expert*4) as u32);let (mut before,mut other)=(0usize,0usize);
                while other<r {let rival=crate::ptx::load(choice+(other*4) as u32);
                    before+=usize::from(rival>value)+usize::from((rival==value)&(other<expert));other+=1;}
                if before<k {crate::ptx::store(slots+(before*4) as u32,expert as f32);}
                expert+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread!=0 {return;}
            let mut sum=0.0f32;
            for slot in 0..k {let id=crate::ptx::load(slots+(slot*4) as u32) as usize;
                output.add(slot).write(id as f32);let score=crate::ptx::load(base+(id*4) as u32);
                output.add(k+slot).write(score);sum+=score;}
            for slot in 0..k {let target=output.add(k+slot);let mut value=*target;if s.taps!=0 {value/=sum+1e-20;}value*=s.bound;target.write(value);}
            return;
        }
    }
    if thread!=0 {return;}
    let width=r/h;
    for group in 0..h {let (mut a,mut b)=(f32::NEG_INFINITY,f32::NEG_INFINITY);
        for expert in group*width..(group+1)*width {let value=crate::ptx::load(choice+(expert*4) as u32);
            if value>a {b=a;a=value;} else if value>b {b=value;}}
        crate::ptx::store(groups+(group*4) as u32,a+b);crate::ptx::store(kept+(group*4) as u32,0.0);}
    for _ in 0..d {let mut id=0;let mut best=f32::NEG_INFINITY;
        for group in 0..h {let value=crate::ptx::load(groups+(group*4) as u32);if value>best {best=value;id=group;}}
        crate::ptx::store(kept+(id*4) as u32,1.0);crate::ptx::store(groups+(id*4) as u32,f32::NEG_INFINITY);}
    let mut sum=0.0f32;
    for slot in 0..k {let mut id=0;let mut best=f32::NEG_INFINITY;
        // Kept groups are immutable here. Hoist their test and the division;
        // their contiguous ranges still visit experts in original order.
        for group in 0..h {if crate::ptx::load(kept+(group*4) as u32)!=0.0 {
            for expert in group*width..(group+1)*width {let value=crate::ptx::load(choice+(expert*4) as u32);
                if value>best {best=value;id=expert;}}
        }}
        output.add(slot).write(id as f32);let score=crate::ptx::load(base+(id*4) as u32);
        output.add(k+slot).write(score);sum+=score;crate::ptx::store(choice+(id*4) as u32,f32::NEG_INFINITY);
    }
    for slot in 0..k {let target=output.add(k+slot);let mut value=*target;if s.taps!=0 {value/=sum+1e-20;}value*=s.bound;target.write(value);}
    return;
}};}
#[no_mangle]
pub extern "ptx-kernel" fn causal_operation(p:Arguments) {resident_causal_operation(p)}
#[inline(always)]
pub(crate) fn resident_causal_operation(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and the exact same-context
    // allocation extents. Every input and weight is immutable through completion.
    // The host completes the private output before this launch: it zeroes it first, except for the operations of
    // causal_ops::writes_every_word, whose every unit writes every word of its slice before any dependent read (host test and
    // device sweep with the output poisoned), so their output starts with whatever the allocation held.
    // It overlaps no input. Each valid thread
    // creates only its own disjoint unit slice. Empty inputs use aligned dangling
    // pointers and zero lengths. All nonempty addresses are aligned and live.
    // Indirect operations are admitted only by the checked row-owner backend:
    // its table entries retain initialized same-context rows through completion.
    // Routing has exactly one block. Its bounded shared tile assigns each
    // score and output scratch word to one thread. Every thread reaches the
    // barrier before thread zero alone reads that tile and writes selections.
    // The one-group form adds k slot words and one flag word behind the tile's
    // other words (checked against the tile size). Thread zero clears the flag
    // before the first barrier; any thread may set it to the same value after
    // that barrier; all read it after the second. With every choice finite the
    // order (choice descending, expert ascending) is total, so each slot word
    // has exactly one writer, and thread zero reads the slots after the third
    // barrier. Each thread reads only initialized score and choice words.
    // Larger valid routes keep the original complete-unit schedule.
    unsafe {
        // Table-form attention (host-validated pointer tables): its own launch shapes, no unit prologue.
        if p.shape.op==crate::causal_ops::Op::ScoresTable {crate::attention_table::scores(p);return;}
        if p.shape.op==crate::causal_ops::Op::ValuesTable {crate::attention_table::values(p);return;}
        if p.shape.op==crate::causal_ops::Op::ScoresPacked {scores_packed(p);return;}
        if p.shape.op==crate::causal_ops::Op::ValuesPacked {values_packed(p);return;}
        let index=crate::index();let Some((lengths,weight,units,words))=p.shape.lengths() else {return;};
        if crate::latent_ops::route_shared_words(p.shape).is_some() {shared_route!(p,words);}
        if index>=units {return;}
        let input=core::array::from_fn(|i|core::slice::from_raw_parts(p.input[i] as *const f32,lengths[i]));
        let weight=core::slice::from_raw_parts(p.weight as *const f32,weight);
        let output=core::slice::from_raw_parts_mut((p.output as *mut f32).add(index*words),words);
        if p.shape.op==crate::causal_ops::Op::ScoresIndirect || p.shape.op==crate::causal_ops::Op::ValuesIndirect {
            let (dim,rows,live,other)=(p.shape.dim as usize,p.shape.rows as usize,p.shape.taps as usize,p.shape.cols as usize);
            let table=p.input[1] as *const u64;
            let value=if p.shape.op==crate::causal_ops::Op::ScoresIndirect {
                let head=index/rows;let slot=index%rows;
                if slot<live {
                    let source=(*table.add(slot)) as *const f32;let mut sum=0.0;
                    for feature in 0..dim {sum+=input[0][head*dim+feature]*(*source.add(head*(dim+other)+feature));}
                    let sum=f32::from_bits(crate::next_math::bf16(sum)<<16);
                    f32::from_bits(crate::next_math::bf16(sum*(1.0/crate::qsa_common::root(dim as f32)))<<16)
                } else {0.0}
            } else {
                let head=index/dim;let feature=index%dim;let mut sum=0.0;
                for slot in 0..live {let source=(*table.add(slot)) as *const f32;
                    sum+=input[0][head*rows+slot]*(*source.add(head*(other+dim)+other+feature));}
                f32::from_bits(crate::next_math::bf16(sum)<<16)
            };
            output[0]=value;return;
        }
        // Grouped plain rows use the one-row project_f32_ordered sequence for each
        // token: mul.rn.f32 of weight and input, then add.rn.f32 in ascending feature
        // order from +0. Pinned instructions keep the compiler from contracting them.
        if p.shape.op==crate::causal_ops::Op::ProjectRows {
            let cols=p.shape.cols as usize;
            for (token,out) in output.iter_mut().enumerate() {
                let mut sum=0.0f32;
                for j in 0..cols {
                    let a=weight[index*cols+j];let b=input[0][token*cols+j];let product: f32;
                    core::arch::asm!("mul.rn.f32 {p}, {a}, {b};",p=out(reg32)product,a=in(reg32)a,b=in(reg32)b);
                    core::arch::asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)sum,p=in(reg32)product);
                }
                *out=sum;
            }
            return;
        }
        // The same chains with BF16 weights packed two to a word: each widened half is the f32 the FP32 copy held.
        if p.shape.op==crate::causal_ops::Op::ProjectRowsBf16 {
            let cols=p.shape.cols as usize;
            for (token,out) in output.iter_mut().enumerate() {
                let mut sum=0.0f32;
                for j in 0..cols {
                    let bits=weight[(index*cols+j)/2].to_bits();let a=if j%2==0 {f32::from_bits(bits<<16)} else {f32::from_bits(bits&0xffff_0000)};
                    let b=input[0][token*cols+j];let product: f32;
                    core::arch::asm!("mul.rn.f32 {p}, {a}, {b};",p=out(reg32)product,a=in(reg32)a,b=in(reg32)b);
                    core::arch::asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)sum,p=in(reg32)product);
                }
                *out=sum;
            }
            return;
        }
        unit(p.shape,index,input,weight,output);
    }
}

/// Packed-row table attention. Same chains as `attention_table::{scores,values}`: each key or value operand is the
/// BF16 pattern of its packed half shifted to the high half word, which is the exact f32 the unpacked row held
/// (every expanded word is BF16-rounded at production). Table entries match the table form, except the address is
/// the packed row base and the stride counts packed words a head ((key_dim+value_dim)/2). Scores reads features
/// 0..dim of the head segment; values reads features cols..cols+dim. Adjacent features share a word (even = low half).
#[inline(always)]
pub(crate) fn scores_packed(p:Arguments) {
    // SAFETY: The host validated the query, the table and every packed row it names (attention_table_proven:
    // dim and cols 256), keeps all owners live through completion, and gives the output heads*rows private words.
    // Thread (block, lane) owns exactly one output word and writes it once; slots past live write 0.0 and read
    // nothing. Key loads stay inside the validated packed span (dim/2 words from head*stride, dim a multiple of 8,
    // so each 16-byte load is whole); the body traps on any other shape or block size.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        let shape=p.shape;let (heads,dim,rows,live)=(shape.heads as usize,shape.dim as usize,shape.rows as usize,shape.taps as usize);
        if _block_dim_x()!=128 || shape.op!=crate::causal_ops::Op::ScoresPacked || dim%8!=0 || shape.cols%2!=0 || live>rows {asm!("trap;");}
        let tiles=rows.div_ceil(128);let block=_block_idx_x() as usize;
        let head=block/tiles;let slot=block%tiles*128+_thread_idx_x() as usize;
        if head>=heads || slot>=rows {return;}
        let output=(p.output as *mut f32).add(head*rows+slot);
        if slot>=live {output.write(0.0);return;}
        let entry=(p.input[1] as *const u32).add(slot*4);
        let address=u64::from(*entry)|u64::from(*entry.add(1))<<32;let stride=*entry.add(2) as usize;
        let key=(address as *const u32).add(head*stride);let query=(p.input[0] as *const f32).add(head*dim);
        let mut sum=0.0f32;let mut at=0;
        // Sixteen features a step: both packed loads and all four query loads issue before the fold, so two
        // 8-feature chunks are in flight at once. The fold still adds features ascending, one mul and one
        // add a feature, same operands as the 8-feature form below (kept for a dim that ends in 8).
        while at+16<=dim {
            // The table scores kernel names the key line 128 bytes ahead to L2; packed, that is 64 features.
            if at%64==0 && at+64<dim {asm!("prefetch.global.L2 [{p}];",p=in(reg64)key.add((at+64)/2),options(nostack,readonly));}
            let (w0,w1,w2,w3):(u32,u32,u32,u32);let (x0,x1,x2,x3):(u32,u32,u32,u32);
            let (q0,q1,q2,q3):(f32,f32,f32,f32);let (q4,q5,q6,q7):(f32,f32,f32,f32);let (q8,q9,q10,q11):(f32,f32,f32,f32);let (q12,q13,q14,q15):(f32,f32,f32,f32);
            asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)w0,b=out(reg32)w1,c=out(reg32)w2,d=out(reg32)w3,pointer=in(reg64)key.add(at/2),options(nostack,readonly));
            asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)x0,b=out(reg32)x1,c=out(reg32)x2,d=out(reg32)x3,pointer=in(reg64)key.add(at/2+4),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q0,b=out(reg32)q1,c=out(reg32)q2,d=out(reg32)q3,pointer=in(reg64)query.add(at),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q4,b=out(reg32)q5,c=out(reg32)q6,d=out(reg32)q7,pointer=in(reg64)query.add(at+4),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q8,b=out(reg32)q9,c=out(reg32)q10,d=out(reg32)q11,pointer=in(reg64)query.add(at+8),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q12,b=out(reg32)q13,c=out(reg32)q14,d=out(reg32)q15,pointer=in(reg64)query.add(at+12),options(nostack,readonly));
            add!(sum,mul!(q0,f32::from_bits(w0<<16)));add!(sum,mul!(q1,f32::from_bits(w0&0xffff_0000)));
            add!(sum,mul!(q2,f32::from_bits(w1<<16)));add!(sum,mul!(q3,f32::from_bits(w1&0xffff_0000)));
            add!(sum,mul!(q4,f32::from_bits(w2<<16)));add!(sum,mul!(q5,f32::from_bits(w2&0xffff_0000)));
            add!(sum,mul!(q6,f32::from_bits(w3<<16)));add!(sum,mul!(q7,f32::from_bits(w3&0xffff_0000)));
            add!(sum,mul!(q8,f32::from_bits(x0<<16)));add!(sum,mul!(q9,f32::from_bits(x0&0xffff_0000)));
            add!(sum,mul!(q10,f32::from_bits(x1<<16)));add!(sum,mul!(q11,f32::from_bits(x1&0xffff_0000)));
            add!(sum,mul!(q12,f32::from_bits(x2<<16)));add!(sum,mul!(q13,f32::from_bits(x2&0xffff_0000)));
            add!(sum,mul!(q14,f32::from_bits(x3<<16)));add!(sum,mul!(q15,f32::from_bits(x3&0xffff_0000)));at+=16;
        }
        while at<dim {
            let (w0,w1,w2,w3):(u32,u32,u32,u32);let (q0,q1,q2,q3):(f32,f32,f32,f32);let (q4,q5,q6,q7):(f32,f32,f32,f32);
            asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)w0,b=out(reg32)w1,c=out(reg32)w2,d=out(reg32)w3,pointer=in(reg64)key.add(at/2),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q0,b=out(reg32)q1,c=out(reg32)q2,d=out(reg32)q3,pointer=in(reg64)query.add(at),options(nostack,readonly));
            asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)q4,b=out(reg32)q5,c=out(reg32)q6,d=out(reg32)q7,pointer=in(reg64)query.add(at+4),options(nostack,readonly));
            add!(sum,mul!(q0,f32::from_bits(w0<<16)));add!(sum,mul!(q1,f32::from_bits(w0&0xffff_0000)));
            add!(sum,mul!(q2,f32::from_bits(w1<<16)));add!(sum,mul!(q3,f32::from_bits(w1&0xffff_0000)));
            add!(sum,mul!(q4,f32::from_bits(w2<<16)));add!(sum,mul!(q5,f32::from_bits(w2&0xffff_0000)));
            add!(sum,mul!(q6,f32::from_bits(w3<<16)));add!(sum,mul!(q7,f32::from_bits(w3&0xffff_0000)));at+=8;
        }
        let sum=f32::from_bits(crate::next_math::bf16(sum)<<16);
        output.write(f32::from_bits(crate::next_math::bf16(sum*(1.0/crate::qsa_common::root(dim as f32)))<<16));
    }
}
#[inline(always)]
pub(crate) fn values_packed(p:Arguments) {
    // SAFETY: As for scores_packed: host-validated probabilities, table and packed rows; output heads*dim private
    // words. Thread (block, lane) owns the F output words head*dim+tile*32*F+lane*F.., each written once. Value
    // feature f of a head sits in packed word cols/2+f/2 of the head segment (cols even), inside the validated span.
    // The S-slot preload only moves loads earlier, as in attention_table::values: each feature still starts at
    // +0.0 and adds every slot in ascending order, one mul.rn.f32 and one add.rn.f32 a slot, same operands.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        const F:usize=crate::causal_ops::PACKED_VALUES_FEATURES as usize;
        const W:usize=(F+1)/2;
        const S:usize=16;
        let shape=p.shape;let (heads,dim,rows,live,keys)=(shape.heads as usize,shape.dim as usize,shape.rows as usize,shape.taps as usize,shape.cols as usize);
        if _block_dim_x()!=32 || shape.op!=crate::causal_ops::Op::ValuesPacked || dim%(32*F)!=0 || keys%2!=0 || live>rows {asm!("trap;");}
        let tiles=dim/(32*F);let block=_block_idx_x() as usize;let head=block/tiles;let first=block%tiles*32*F+_thread_idx_x() as usize*F;
        if head>=heads {return;}
        let table=p.input[1] as *const u32;let probabilities=(p.input[0] as *const f32).add(head*rows);
        let word0=first/2;let mut sum=[0.0f32;F];let mut slot=0;
        while slot+S<=live {
            let mut weights=[0.0f32;S];let mut words=[[0u32;W];S];let mut next=0;
            while next<S {
                let entry=table.add((slot+next)*4);let address=u64::from(*entry)|u64::from(*entry.add(1))<<32;let stride=*entry.add(2) as usize;
                let source=(address as *const u32).add(head*stride+keys/2+word0);weights[next]=*probabilities.add(slot+next);
                let mut w=0;while w<W {words[next][w]=*source.add(w);w+=1;}
                next+=1;
            }
            next=0;while next<S {
                let mut e=0;while e<F {let word=words[next][(first+e)/2-word0];add!(sum[e],mul!(weights[next],f32::from_bits(if (first+e)%2==0 {word<<16} else {word&0xffff_0000})));e+=1;}
                next+=1;
            }
            slot+=S;
        }
        while slot<live {
            let entry=table.add(slot*4);let address=u64::from(*entry)|u64::from(*entry.add(1))<<32;let stride=*entry.add(2) as usize;
            let source=(address as *const u32).add(head*stride+keys/2+word0);let weight=*probabilities.add(slot);
            let mut words=[0u32;W];let mut w=0;while w<W {words[w]=*source.add(w);w+=1;}
            let mut e=0;while e<F {let word=words[(first+e)/2-word0];add!(sum[e],mul!(weight,f32::from_bits(if (first+e)%2==0 {word<<16} else {word&0xffff_0000})));e+=1;}
            slot+=1;
        }
        let output=(p.output as *mut f32).add(head*dim+first);
        let mut e=0;while e<F {output.add(e).write(f32::from_bits(crate::next_math::bf16(sum[e])<<16));e+=1;}
    }
}

/// The warp/block-per-unit versions (`causal_fast.rs`) have their own entry, so their registers and shared memory do not
/// set the occupancy of the generic kernel above, which also runs large copies of other lanes. The host launches it only for
/// shapes `causal_ops::fast_admitted` accepts, with `causal_ops::geometry`; any other op traps.
#[no_mangle]
pub extern "ptx-kernel" fn causal_fast(p:Arguments) {resident_causal_fast(&p)}
#[inline(always)]
pub(crate) fn resident_causal_fast(p:&Arguments) {
    // SAFETY: The host validated Shape::lengths and the exact same-context extents before this launch, and chose this
    // entry only for an operation and block size that fast_admitted accepts; anything else traps here. Each version
    // states its own extents and barrier discipline. The caller retains the immutable
    // arguments through this call. No reference escapes.
    unsafe {
        if !crate::causal_ops::fast_admitted(p.shape,core::arch::nvptx::_block_dim_x()) {core::arch::nvptx::trap();}
        match p.shape.op {
            crate::causal_ops::Op::Prepare=>crate::causal_fast::prepare(*p),
            crate::causal_ops::Op::GatedNorm=>crate::causal_fast::gated_norm(*p),
            crate::causal_ops::Op::PoolScores=>crate::causal_fast::pool_scores(*p),
            crate::causal_ops::Op::Step|crate::causal_ops::Op::StepColumns=>crate::causal_fast::step_columns_128(p),
            crate::causal_ops::Op::StepPlanar=>crate::causal_fast::step_columns_128(p),
            crate::causal_ops::Op::ConvParts=>crate::causal_small::conv_parts(*p),
            crate::causal_ops::Op::Conv=>crate::causal_small::conv(*p),
            crate::causal_ops::Op::KeyNorm=>crate::causal_small::key_norm(*p),
            crate::causal_ops::Op::Mxfp8Quantize=>crate::causal_small::mxfp8_quantize(*p,false),
            crate::causal_ops::Op::Mxfp8QuantizePositiveZero=>crate::causal_small::mxfp8_quantize(*p,true),
            crate::causal_ops::Op::Softmax=>crate::causal_fast::softmax::<false>(*p,0,0),
            crate::causal_ops::Op::Gather=>crate::causal_fast::gather(*p),
            crate::causal_ops::Op::PoolChannels=>crate::causal_fast::pool_channels(*p),
            _=>crate::causal_fast::step_layout_128(*p),
        }
    }
}
