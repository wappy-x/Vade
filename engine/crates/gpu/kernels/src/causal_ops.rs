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
    let mut expert=thread;
    while expert<r {let score=crate::latent_ops::sigmoid(*input.add(expert));
        crate::ptx::store(base+(expert*4) as u32,score);
        crate::ptx::store(choice+(expert*4) as u32,score+*weight.add(expert));expert+=threads;}
    let mut word=2*k+thread;while word<$words {output.add(word).write(0.0);word+=threads;}
    core::arch::nvptx::_syncthreads();
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
    // Larger valid routes keep the original complete-unit schedule.
    unsafe {
        // Table-form attention (host-validated pointer tables): its own launch shapes, no unit prologue.
        if p.shape.op==crate::causal_ops::Op::ScoresTable {crate::attention_table::scores(p);return;}
        if p.shape.op==crate::causal_ops::Op::ValuesTable {crate::attention_table::values(p);return;}
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
        unit(p.shape,index,input,weight,output);
    }
}

/// The warp/block-per-unit versions (`causal_fast.rs`) have their own entry, so their registers and shared memory do not
/// set the occupancy of the generic kernel above, which also runs large copies of other lanes. The host launches it only for
/// shapes `causal_ops::fast_admitted` accepts, with `causal_ops::geometry`; any other op traps.
#[no_mangle]
pub extern "ptx-kernel" fn causal_fast(p:Arguments) {resident_causal_fast(p)}
#[inline(always)]
pub(crate) fn resident_causal_fast(p:Arguments) {
    // SAFETY: The host validated Shape::lengths and the exact same-context extents before this launch, and chose this
    // entry only for an operation and block size that fast_admitted accepts; anything else traps here. Each version
    // states its own extents and barrier discipline.
    unsafe {
        if !crate::causal_ops::fast_admitted(p.shape,core::arch::nvptx::_block_dim_x()) {core::arch::nvptx::trap();}
        match p.shape.op {
            crate::causal_ops::Op::Prepare=>crate::causal_fast::prepare(p),
            crate::causal_ops::Op::GatedNorm=>crate::causal_fast::gated_norm(p),
            crate::causal_ops::Op::PoolScores=>crate::causal_fast::pool_scores(p),
            crate::causal_ops::Op::Step|crate::causal_ops::Op::StepColumns=>crate::causal_fast::step_columns_128(p),
            crate::causal_ops::Op::StepPlanar=>crate::causal_fast::step_columns_128(p),
            crate::causal_ops::Op::ConvParts=>crate::causal_small::conv_parts(p),
            crate::causal_ops::Op::Conv=>crate::causal_small::conv(p),
            crate::causal_ops::Op::KeyNorm=>crate::causal_small::key_norm(p),
            crate::causal_ops::Op::Mxfp8Quantize=>crate::causal_small::mxfp8_quantize(p,false),
            crate::causal_ops::Op::Mxfp8QuantizePositiveZero=>crate::causal_small::mxfp8_quantize(p,true),
            crate::causal_ops::Op::Softmax=>crate::causal_fast::softmax::<false>(p,0,0),
            crate::causal_ops::Op::Gather=>crate::causal_fast::gather(p),
            crate::causal_ops::Op::PoolChannels=>crate::causal_fast::pool_channels(p),
            _=>crate::causal_fast::step_layout_128(p),
        }
    }
}
