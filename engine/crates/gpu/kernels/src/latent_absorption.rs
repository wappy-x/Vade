//! SUM-ORDER v1 latent absorption with fixed strip ownership.
use crate::causal_ops::{Arguments,Op};
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption(p:Arguments) {resident_latent_absorption(p)}
#[inline(always)]
pub(crate) fn resident_latent_absorption(p:Arguments) {
    // SAFETY: The causal backend validates all input and packed-weight extents in Shape::lengths,
    // retains them on the same ordered stream, and gives this launch a disjoint private output.
    // The backend launches 128 threads, for AbsorbQuery and AbsorbProject only; any other operation traps.
    // Each whole warp below units owns one output word.
    // All lanes execute each shuffle. Lanes read only their in-range terms. Lane zero alone
    // writes the output. Reordered weights hold the exact BF16 checkpoint patterns.
    // Inline arithmetic uses FP32 register operands with gradual underflow and no contraction.
    unsafe {
        use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x}};
        let s=p.shape;let (h,d,v,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize);
        if _block_dim_x()!=128 {asm!("trap;");}let thread=_thread_idx_x() as usize;let lane=thread%32;
        let index=_block_idx_x() as usize*4+thread/32;
        // Scores and values have their own kernels below. Any other operation here is a host fault: stop.
        let (head,feature,n,units)=match s.op {
            Op::AbsorbQuery=>(index/k,index%k,d,h*k),
            Op::AbsorbProject=>(index/v,index%v,k,h*v),
            _=>{asm!("trap;");return;},
        };if index>=units {return;}
        let x=p.input[0] as *const f32;let w=p.weight as *const u16;
        let mut sum=0.0f32;let mut first=lane*8;
        while first<n {for offset in 0..8 {let j=first+offset;if j<n {
            let (a,b)=if s.op==Op::AbsorbQuery {(*x.add(head*d+j),f32::from_bits((*w.add(head*(d+v)*k+feature*d+j) as u32)<<16))}
                else {(*x.add(head*k+j),f32::from_bits((*w.add((head*(d+v)+d+feature)*k+j) as u32)<<16))};
            let product:f32;
            asm!("mul.rn.f32 {v}, {a}, {b};",v=out(reg32)product,a=in(reg32)a,b=in(reg32)b,options(nomem,nostack));
            asm!("add.rn.f32 {v}, {v}, {p};",v=inout(reg32)sum,p=in(reg32)product,options(nomem,nostack));
        }}first+=256;}
        for distance in [16u32,8,4,2,1] {let other:f32;
            asm!("shfl.sync.down.b32 {o}, {s}, {d}, 31, -1;",o=out(reg32)other,s=in(reg32)sum,d=in(reg32)distance,options(nomem,nostack));
            if lane<distance as usize {asm!("add.rn.f32 {s}, {s}, {o};",s=inout(reg32)sum,o=in(reg32)other,options(nomem,nostack));}
        }
        if lane==0 {(p.output as *mut f32).add(index).write(crate::latent_absorb::round(sum));}
    }
}

/// Four complete heads share each latent load. Each warp is one strip, with lanes over features.
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption_values(p:Arguments) {resident_latent_absorption_values(p)}
#[inline(always)]
pub(crate) fn resident_latent_absorption_values(p:Arguments) {
    // SAFETY: The backend admits only AbsorbValues and validates heads*rows probabilities,
    // rows*cols latent values, one live count, and a disjoint heads*cols output. It launches
    // 1024 threads, one warp per strip, in ceil(heads/4)*ceil(cols/32) blocks. The count is
    // produced by validated selection and is at most rows. All threads reach every barrier.
    // Each (head-in-tile, strip, feature) owns one shared word in 16384 bytes. A join reads
    // the previous barrier's upper strip and writes only its own lower strip. Strip zero
    // alone writes each in-range output. Register arithmetic uses separate RN operations.
    unsafe {
        use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x,_syncthreads}};
        if _block_dim_x()!=1024 || p.shape.op!=Op::AbsorbValues {asm!("trap;");}
        let s=p.shape;let (heads,rows,cols)=(s.heads as usize,s.rows as usize,s.cols as usize);
        let count=*(p.input[2] as *const f32) as usize;if count>rows {asm!("trap;");}
        let thread=_thread_idx_x() as usize;let strip=thread/32;let lane=thread%32;
        let tiles=cols.div_ceil(32);let block=_block_idx_x() as usize;
        let first_head=block/tiles*4;let feature=block%tiles*32+lane;
        let prob=p.input[0] as *const f32;let latent=p.input[1] as *const f32;
        let mut sums=[0.0f32;4];let mut first=strip*8;
        while first<count {for offset in 0..8 {let token=first+offset;if token<count && feature<cols {
            let x=*latent.add(token*cols+feature);
            for (head,sum) in sums.iter_mut().enumerate() {if first_head+head<heads {let weight=*prob.add((first_head+head)*rows+token);let product:f32;
                asm!("mul.rn.f32 {p}, {w}, {x};",p=out(reg32)product,w=in(reg32)weight,x=in(reg32)x,options(nomem,nostack));
                asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)*sum,p=in(reg32)product,options(nomem,nostack));
            }}
        }}first+=256;}
        let base=crate::ptx::shared::<16384>();
        for (head,&sum) in sums.iter().enumerate() {crate::ptx::store(base+((head*1024+thread)*4) as u32,sum);}
        _syncthreads();
        for distance in [16usize,8,4,2,1] {
            if strip<distance {for (head,sum) in sums.iter_mut().enumerate() {
                let other=crate::ptx::load(base+((head*1024+thread+distance*32)*4) as u32);
                asm!("add.rn.f32 {s}, {s}, {o};",s=inout(reg32)*sum,o=in(reg32)other,options(nomem,nostack));
                crate::ptx::store(base+((head*1024+thread)*4) as u32,*sum);
            }}_syncthreads();
        }
        if strip==0 && feature<cols {for (head,sum) in sums.into_iter().enumerate() {if first_head+head<heads {
            (p.output as *mut f32).add((first_head+head)*cols+feature).write(crate::latent_absorb::round(sum));
        }}}
    }
}

/// One warp computes four heads of one token, sharing every latent load.
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption_scores(p:Arguments) {resident_latent_absorption_scores(p)}
#[inline(always)]
pub(crate) fn resident_latent_absorption_scores(p:Arguments) {
    // SAFETY: The backend validates heads*cols absorbed queries, rows*cols latent values,
    // one live count, and a disjoint heads*rows output. It launches 128 threads and
    // ceil(ceil(heads/4)*rows/4) blocks. Each whole warp owns one token for four heads.
    // All participating lanes reach each shuffle. Bounds exclude absent heads and rows.
    // Lane zero alone writes the owned outputs. Register-only RN arithmetic cannot alias memory.
    unsafe {
        use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x}};
        if _block_dim_x()!=128 || p.shape.op!=Op::AbsorbScores {asm!("trap;");}
        let s=p.shape;let (heads,rows,cols)=(s.heads as usize,s.rows as usize,s.cols as usize);
        let thread=_thread_idx_x() as usize;let lane=thread%32;let unit=_block_idx_x() as usize*4+thread/32;
        let head0=unit/rows*4;let token=unit%rows;if head0>=heads {return;}
        let count=*(p.input[2] as *const f32) as usize;let out=p.output as *mut f32;
        if token>=count {if lane==0 {for head in 0..4 {if head0+head<heads {out.add((head0+head)*rows+token).write(0.0);}}}return;}
        let query=p.input[0] as *const f32;let latent=(p.input[1] as *const f32).add(token*cols);
        let mut sums=[0.0f32;4];let mut first=lane*8;
        while first<cols {for offset in 0..8 {let j=first+offset;if j<cols {let x=*latent.add(j);
            for (head,sum) in sums.iter_mut().enumerate() {if head0+head<heads {let weight=*query.add((head0+head)*cols+j);let product:f32;
                asm!("mul.rn.f32 {p}, {w}, {x};",p=out(reg32)product,w=in(reg32)weight,x=in(reg32)x,options(nomem,nostack));
                asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)*sum,p=in(reg32)product,options(nomem,nostack));
            }}
        }}first+=256;}
        for distance in [16u32,8,4,2,1] {for sum in &mut sums {let other:f32;
            asm!("shfl.sync.down.b32 {o}, {s}, {d}, 31, -1;",o=out(reg32)other,s=in(reg32)*sum,d=in(reg32)distance,options(nomem,nostack));
            if lane<distance as usize {asm!("add.rn.f32 {s}, {s}, {o};",s=inout(reg32)*sum,o=in(reg32)other,options(nomem,nostack));}
        }}
        if lane==0 {let scale=1.0/crate::causal_ops::root(s.dim as f32);for (head,sum) in sums.into_iter().enumerate() {if head0+head<heads {
            let value:f32;
            asm!("mul.rn.f32 {v}, {a}, {b};",v=out(reg32)value,a=in(reg32)sum,b=in(reg32)scale,options(nomem,nostack));
            out.add((head0+head)*rows+token).write(value);
        }}}
    }
}
