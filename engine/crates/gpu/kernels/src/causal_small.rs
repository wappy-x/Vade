//! Device versions of two small single-token causal operations that the generic one-thread-per-unit path runs poorly.
//! Each writes the same bytes as the generic code (`causal_ops::unit`, `latent_ops::unit`): the same expressions in the
//! same order, no fused multiply-add added, every serial sum kept as one serial chain. Chosen by `causal_ops::fast_admitted`;
//! the caller guarantees the block has 128 threads and every thread of the block enters (no early exit before a barrier).
use crate::causal_ops::{Arguments,round,sigmoid,root};

/// Shared words `key_norm` needs: the staged key (at most 1,024 words), then the mean and the inverse norm.
pub(crate) const KEY_NORM_SHARED_BYTES:usize=(1024+4)*4;

/// Op::Conv, one thread per (plane, channel). The generic Op::Conv is one thread per plane (three threads in all); the
/// split ConvChannels plus ConvLayout pair exists only to get parallelism and moves every word twice. This writes the
/// final layout directly: plane p, tap t, channel c lands at `p*taps*width + t*width + c`, exactly where `unit` puts it
/// (tap 0 the convolution, tap t+1 the history shifted by one). The sum over taps is the same ascending chain from +0.
pub(crate) fn conv(p:Arguments) {
    // SAFETY: The host checked Shape::lengths and the exact same-context extents: x0 holds 3*width values, x1
    // 3*(taps-1)*width, the weight 3*width*taps, the output 3*width*taps; the output overlaps no input. Every index
    // below is inside those extents (plane < 3, channel < width, tap < taps). Each (plane, channel) has one writer
    // for its taps output words. There is no barrier, so the grid-stride loop may leave at any point.
    unsafe {
        let s=p.shape;let (heads,d,taps)=(s.heads as usize,s.dim as usize,s.taps as usize);let width=heads*d;
        let (x0,x1,w)=(p.input[0] as *const f32,p.input[1] as *const f32,p.weight as *const f32);let out=p.output as *mut f32;
        let history=(taps-1)*width;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*128+core::arch::nvptx::_thread_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*128;
        while index<3*width {
            let plane=index/width;let ch=index%width;
            let mut sum=0.0f32;
            let mut tap=0;while tap<taps {
                let v=if tap+1==taps {*x0.add(plane*width+ch)} else {*x1.add(plane*history+tap*width+ch)};
                sum+=v**w.add((plane*width+ch)*taps+tap);tap+=1;
            }
            let base=out.add(plane*taps*width+ch);
            *base=round(sum*sigmoid(sum));
            tap=0;while tap+1<taps {
                *base.add((tap+1)*width)=if tap+2==taps {*x0.add(plane*width+ch)} else {*x1.add(plane*history+(tap+1)*width+ch)};tap+=1;
            }
            index+=stride;
        }
    }
}

/// Op::KeyNorm, one block of 128 threads. The key's mean and squares are two serial chains in ascending order from +0
/// (that order is the model's, so it stays one chain); the generic code runs them in one thread straight from global
/// memory, one dependent load after another. Here the 128 threads first stage the key in shared memory, thread 0 runs the
/// same two chains from shared memory, and every thread then writes its own output words with the same expression.
pub(crate) fn key_norm(p:Arguments) {
    // SAFETY: The host checked Shape::lengths and the exact same-context extents: x0 holds dim values, the weight 2*dim,
    // the output dim; the output overlaps no input. dim is at most 1,024, so the staged key and the two result words fit
    // in KEY_NORM_SHARED_BYTES. The launch is one block of exactly 128 threads and every thread reaches both barriers.
    // Thread 0 alone writes the two result words before the second barrier; every other shared read comes after it.
    unsafe {
        let s=p.shape;let d=s.dim as usize;
        let (x,w)=(p.input[0] as *const f32,p.weight as *const f32);let out=p.output as *mut f32;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        let base=crate::ptx::shared::<KEY_NORM_SHARED_BYTES>();
        let mut j=tid;while j<d {crate::ptx::store(base+(j*4) as u32,*x.add(j));j+=128;}
        crate::ptx::sync();
        if tid==0 {
            let mut sum=0.0f32;let mut i=0;while i<d {sum+=crate::ptx::load(base+(i*4) as u32);i+=1;}
            let mean=sum/d as f32;
            let mut squares=0.0f32;i=0;while i<d {let a=crate::ptx::load(base+(i*4) as u32)-mean;squares+=a*a;i+=1;}
            let inv=1.0/root(squares/d as f32+s.epsilon);
            crate::ptx::store(base+(d*4) as u32,mean);crate::ptx::store(base+(d*4+4) as u32,inv);
        }
        crate::ptx::sync();
        let mean=crate::ptx::load(base+(d*4) as u32);let inv=crate::ptx::load(base+(d*4+4) as u32);
        j=tid;while j<d {*out.add(j)=round(((crate::ptx::load(base+(j*4) as u32)-mean)*inv)**w.add(j)+*w.add(d+j));j+=128;}
    }
}

/// Op::Mxfp8Quantize and Op::Mxfp8QuantizePositiveZero, one warp per group of 32. The generic code makes every lane read the
/// whole group, round it and search the scale again, 32 times over. The group maximum is the same value for every lane, and a
/// maximum of non-NaN values is exact in any order, so the warp builds it with a butterfly and every lane then runs the
/// unchanged remainder (`mxfp8_quantize::finish`) on that maximum. A group that holds an infinity or NaN after the BF16
/// rounding gives NaN in all 32 lanes, as the generic code does.
pub(crate) fn mxfp8_quantize(p:Arguments,positive_zero:bool) {
    // SAFETY: The host checked Shape::lengths and the exact same-context extents: the input and the output hold cols words
    // (cols is a multiple of 32), the plane is rows (at most 2), the output overlaps no input. Every word index below is
    // group*32+lane with group*32<cols. Each word has one writer. All 32 lanes of a warp take the same trip count (the
    // loop bound depends on the group only), so the vote and the shuffles are converged.
    unsafe {
        let s=p.shape;let (k,plane)=(s.cols as usize,s.rows);
        let (x,out)=(p.input[0] as *const f32,p.output as *mut f32);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut group=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        while group*32<k {
            let i=group*32+lane;let raw=*x.add(i);let v=crate::bf16::rounded(raw);
            let bad=crate::ptx::ballot(!v.is_finite())!=0;
            let mut maximum=v.abs();
            let mut distance=16u32;
            while distance!=0 {
                let other:f32;core::arch::asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)other,v=in(reg32)maximum,d=in(reg32)distance,options(nomem,nostack));
                maximum=maximum.max(other);distance/=2;
            }
            *out.add(i)=if bad {f32::NAN} else {crate::mxfp8_quantize::finish(maximum,raw,plane,positive_zero)};
            group+=stride;
        }
    }
}

/// Op::ConvParts: Conv with the q, k and v inputs apart (no packing copy before the launch) and the outputs in two runs: the
/// three convolution outputs (plane p at p*width + channel), then the three planes' shifted histories (plane p, tap t, channel c at
/// 3*width + p*(taps-1)*width + t*width + c). The decoder takes two views of this one buffer instead of joining the per-plane
/// pieces. Same expressions and the same ascending tap chain from +0 as `unit` of Conv.
pub(crate) fn conv_parts(p:Arguments) {
    // SAFETY: The host checked Shape::lengths and the exact same-context extents: x0, x1 and x2 hold width values each, the
    // history 3*(taps-1)*width, the weight 3*width*taps, the output 3*width*taps; the output overlaps no input. Every index below is
    // inside those extents (plane < 3, channel < width, tap < taps). Each (plane, channel) has one writer for its taps output
    // words, spread over the two runs. There is no barrier, so the grid-stride loop may leave at any point.
    unsafe {
        let s=p.shape;let (heads,d,taps)=(s.heads as usize,s.dim as usize,s.taps as usize);let width=heads*d;
        let parts=[p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32];
        let (hist,w)=(p.input[3] as *const f32,p.weight as *const f32);let out=p.output as *mut f32;
        let history=(taps-1)*width;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*128+core::arch::nvptx::_thread_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*128;
        while index<3*width {
            let plane=index/width;let ch=index%width;let source=parts[plane];
            let mut sum=0.0f32;
            let mut tap=0;while tap<taps {
                let v=if tap+1==taps {*source.add(ch)} else {*hist.add(plane*history+tap*width+ch)};
                sum+=v**w.add((plane*width+ch)*taps+tap);tap+=1;
            }
            *out.add(plane*width+ch)=round(sum*sigmoid(sum));
            let shifted=out.add(3*width+plane*history+ch);
            tap=0;while tap+1<taps {
                *shifted.add(tap*width)=if tap+2==taps {*source.add(ch)} else {*hist.add(plane*history+(tap+1)*width+ch)};tap+=1;
            }
            index+=stride;
        }
    }
}
