//! Deferred post/pre boundaries. Arithmetic follows the resident primitives.
use crate::{mhc_params::{Boundary,MHC_BLOCK_TOKENS},mhc_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_project(p:Boundary) {
    // SAFETY: The caller validates disjoint planes and retains their owners.
    // A block owns one token. Each thread writes distinct residual and
    // normalized words. All threads reach the barriers before dependent reads.
    // Warp zero publishes the inverse in one shared word. Each complete warp
    // owns distinct coefficient rows. All lanes reach every full-mask shuffle,
    // including for partial strips. Only lane zero writes each coefficient.
    // Previous planes are read only when present, within their checked extents.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
        if threads<64 || threads%32!=0 {core::arch::asm!("trap;");}
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let residual=(p.residual as *mut f32).add(token*wide);let mut feature=thread;
        while feature<wide {
            let value=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                else {crate::mhc_boundary_math::combine(
                    slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                    slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                    crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature%hidden)),
                    streams,hidden,feature/hidden,feature%hidden)};
            *residual.add(feature)=value;feature+=threads;
        }
        let shared=crate::ptx::shared::<4>();
        core::arch::nvptx::_syncthreads();
        if thread<32 {
            let squares=crate::tree_reduction::warp_fold!(wide,lane,|column|{let value=*residual.add(column);value*value});
            if lane==0 {crate::ptx::store(shared,1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps));}
        }
        core::arch::nvptx::_syncthreads();
        let inverse=crate::ptx::load(shared);let mut feature=thread;
        while feature<wide {*(p.mixing_normalized as *mut f32).add(token*wide+feature)=*residual.add(feature)*inverse;feature+=threads;}
        let mut coefficient=thread/32;
        while coefficient<count {
            let row=(p.weights as *const f32).add(coefficient*wide);
            let total=crate::tree_reduction::warp_fold!(wide,lane,|column|{let normalized=*residual.add(column)*inverse;*row.add(column)*normalized});
            if lane==0 {
                let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
            }
            coefficient+=threads/32;
        }
    }
}

macro_rules! boundary_finish_token {($arg:expr,$tok:expr)=>{{let p=$arg;let token:usize=$tok;
    // SAFETY: The caller supplies validated, live, disjoint planes. The preceding
    // phase on the same stream completes the raw coefficient rows for `token`.
    // All threads of the block reach every barrier and complete warps reach
    // every shuffle. Up to 1024 coefficients and 32 row/inverse scratch values
    // fit the private shared allocation; larger shapes retain the scalar fold.
    // Rows and columns synchronize after each unchanged ordered sum and divide.
    // Activation ends before collapse, collapse before partials, and partials
    // before the unchanged per-lane final norm tree. Each output has one writer;
    // all slice extents and partial indices follow the validated shape.
    {
        let shape=p.shape;
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        const COEFFICIENT_CAPACITY:usize=1024;
        let shared=crate::ptx::shared::<{(COEFFICIENT_CAPACITY+32)*4}>();
        macro_rules! load {($index:expr)=>{crate::ptx::load(shared+(($index)*4) as u32)}}
        macro_rules! store {($index:expr,$value:expr)=>{crate::ptx::store(shared+(($index)*4) as u32,$value)}}
        if streams*streams<=32 {
            if thread<32 {
                let coefficients=(p.coefficients as *mut f32).add(token*count);
                let pre=if lane<streams {math::sigmoid(*coefficients.add(lane))+shape.eps} else {0.0};
                let post=if lane<streams {2.0*math::sigmoid(*coefficients.add(streams+lane))} else {0.0};
                let mut value=if lane<streams*streams {*coefficients.add(2*streams+lane)} else {0.0};
                let row=(lane/streams).min(streams-1);let column=lane%streams;
                macro_rules! read_lane {($source:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^($source)) as u32,31))}}
                if streams==4 {
                    macro_rules! read_mask {($mask:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),$mask,31))}}
                    let along=[(column^0) as u32,(column^1) as u32,(column^2) as u32,(column^3) as u32];
                    let down=[((row^0)*4) as u32,((row^1)*4) as u32,((row^2)*4) as u32,((row^3)*4) as u32];
                    let mut maximum=f32::NEG_INFINITY;
                    crate::strip_warp::unroll!(4,source,{maximum=maximum.max(read_mask!(along[source]));});
                    value=crate::qsa_common::exponential(value-maximum);
                    let mut sum=0.0f32;
                    crate::strip_warp::unroll!(4,source,{sum+=read_mask!(along[source]);});
                    value=value/sum+shape.eps;
                    let mut sum=0.0f32;
                    crate::strip_warp::unroll!(4,source,{sum+=read_mask!(down[source]);});
                    value/=sum+shape.eps;
                    for _iteration in 1..shape.iterations {
                        let mut sum=0.0f32;crate::strip_warp::unroll!(4,source,{sum+=read_mask!(along[source]);});
                        value/=sum+shape.eps;
                        let mut sum=0.0f32;crate::strip_warp::unroll!(4,source,{sum+=read_mask!(down[source]);});
                        value/=sum+shape.eps;
                    }
                } else {
                    let mut maximum=f32::NEG_INFINITY;
                    for source in 0..streams {maximum=maximum.max(read_lane!(row*streams+source));}
                    value=crate::qsa_common::exponential(value-maximum);
                    let mut sum=0.0f32;
                    for source in 0..streams {sum+=read_lane!(row*streams+source);}
                    value=value/sum+shape.eps;
                    let mut sum=0.0f32;
                    for source in 0..streams {sum+=read_lane!(source*streams+column);}
                    value/=sum+shape.eps;
                    for _iteration in 1..shape.iterations {
                        let mut sum=0.0f32;
                        for source in 0..streams {sum+=read_lane!(row*streams+source);}
                        value/=sum+shape.eps;
                        let mut sum=0.0f32;
                        for source in 0..streams {sum+=read_lane!(source*streams+column);}
                        value/=sum+shape.eps;
                    }
                }
                if lane<streams {*coefficients.add(lane)=pre;*coefficients.add(streams+lane)=post;}
                if lane<streams*streams {*coefficients.add(2*streams+lane)=value;}
            }
        } else if count<=COEFFICIENT_CAPACITY {
            let mut coefficient=thread;
            while coefficient<count {store!(coefficient,*(p.coefficients as *const f32).add(token*count+coefficient));coefficient+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread<streams {
                store!(thread,math::sigmoid(load!(thread))+shape.eps);
                store!(streams+thread,2.0*math::sigmoid(load!(streams+thread)));
                let mut maximum=f32::NEG_INFINITY;
                for column in 0..streams {maximum=maximum.max(load!(2*streams+thread*streams+column));}
                store!(COEFFICIENT_CAPACITY+thread,maximum);
            }
            core::arch::nvptx::_syncthreads();
            let mut cell=thread;while cell<streams*streams {let row=cell/streams;let value=load!(2*streams+cell)-load!(COEFFICIENT_CAPACITY+row);
                store!(2*streams+cell,crate::qsa_common::exponential(value));cell+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread<streams {let mut sum=0.0f32;for column in 0..streams {sum+=load!(2*streams+thread*streams+column);}store!(COEFFICIENT_CAPACITY+thread,sum);}
            core::arch::nvptx::_syncthreads();
            let mut cell=thread;while cell<streams*streams {let row=cell/streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+row)+shape.eps);cell+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread<streams {let mut sum=0.0f32;for source in 0..streams {sum+=load!(2*streams+source*streams+thread);}store!(COEFFICIENT_CAPACITY+thread,sum+shape.eps);}
            core::arch::nvptx::_syncthreads();
            let mut cell=thread;while cell<streams*streams {let column=cell%streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+column));cell+=threads;}
            core::arch::nvptx::_syncthreads();
            for _iteration in 1..shape.iterations {
                if thread<streams {let mut sum=0.0f32;for column in 0..streams {sum+=load!(2*streams+thread*streams+column);}store!(COEFFICIENT_CAPACITY+thread,sum+shape.eps);}
                core::arch::nvptx::_syncthreads();
                let mut cell=thread;while cell<streams*streams {let row=cell/streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+row));cell+=threads;}
                core::arch::nvptx::_syncthreads();
                if thread<streams {let mut sum=0.0f32;for source in 0..streams {sum+=load!(2*streams+source*streams+thread);}store!(COEFFICIENT_CAPACITY+thread,sum+shape.eps);}
                core::arch::nvptx::_syncthreads();
                let mut cell=thread;while cell<streams*streams {let column=cell%streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+column));cell+=threads;}
                core::arch::nvptx::_syncthreads();
            }
            let mut coefficient=thread;
            while coefficient<count {*(p.coefficients as *mut f32).add(token*count+coefficient)=load!(coefficient);coefficient+=threads;}
        } else if thread==0 {math::activate(shape,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(token*count),count));}
        core::arch::nvptx::_syncthreads();
        let warps=threads/32;let chunks=hidden.div_ceil(128);
        if streams<=4 && hidden%128==0 && chunks<=warps*4 {
            let warp=thread/32;let mut pre=[0.0f32;4];
            crate::strip_warp::unroll!(4,src,{if src<streams {pre[src]=*(p.coefficients as *const f32).add(token*count+src);}});
            let mut held=[0.0f32;16];
            crate::strip_warp::unroll!(4,wave,{let chunk=warp+wave*warps;if chunk<chunks {
                let mut x=[[0.0f32;4];4];
                crate::strip_warp::unroll!(4,src,{if src<streams {crate::strip_warp::unroll!(4,e,{
                    x[src][e]=*(p.residual as *const f32).add(token*wide+src*hidden+chunk*128+lane+32*e);
                });}});
                let mut value=[0.0f32;4];
                crate::strip_warp::unroll!(4,src,{if src<streams {crate::strip_warp::unroll!(4,e,{
                    crate::strip_warp::add!(value[e],crate::strip_warp::mul!(pre[src],x[src][e]));
                });}});
                let mut squares=0.0f32;
                crate::strip_warp::unroll!(4,e,{
                    let value=crate::bf16::rounded(value[e]);held[wave*4+e]=value;
                    if p.collapsed!=0 {*(p.collapsed as *mut f32).add(token*hidden+chunk*128+lane+32*e)=value;}
                    crate::strip_warp::add!(squares,crate::strip_warp::mul!(value,value));
                });
                let mut distance=16;
                while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                if lane==0 {
                    if p.norm_partials!=0 {*(p.norm_partials as *mut f32).add(token*chunks+chunk)=squares;}
                    else {store!(COEFFICIENT_CAPACITY+chunk,squares);}
                }
            }});
            core::arch::nvptx::_syncthreads();
            if thread<32 {
                let mut squares=0.0f32;let mut chunk=lane;
                while chunk<chunks {
                    squares+=if p.norm_partials!=0 {*(p.norm_partials as *const f32).add(token*chunks+chunk)} else {load!(COEFFICIENT_CAPACITY+chunk)};
                    chunk+=32;
                }
                let mut distance=16;
                while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                store!(COEFFICIENT_CAPACITY+lane,1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs()));
            }
            core::arch::nvptx::_syncthreads();
            let inverse=load!(COEFFICIENT_CAPACITY+lane);
            crate::strip_warp::unroll!(4,wave,{let chunk=warp+wave*warps;if chunk<chunks {
                crate::strip_warp::unroll!(4,e,{let feature=chunk*128+lane+32*e;
                    let normalized=crate::layer::round_bf16(held[wave*4+e]*inverse*1.0);
                    let value=normalized * *(p.norm_weight as *const f32).add(feature);
                    *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
                });
            }});
        } else {
            let mut feature=thread;
            while feature<hidden {
                *(p.collapsed as *mut f32).add(token*hidden+feature)=crate::mhc_boundary_math::collapse(
                    slice::from_raw_parts((p.residual as *const f32).add(token*wide),wide),
                    slice::from_raw_parts((p.coefficients as *const f32).add(token*count),streams),streams,hidden,feature);
                feature+=threads;
            }
            core::arch::nvptx::_syncthreads();
            let chunks=hidden.div_ceil(128);let mut chunk=thread/32;
            while chunk<chunks {
                let mut squares=0.0f32;let mut feature=chunk*128+lane;
                while feature<(chunk*128+128).min(hidden) {
                    let value=*(p.collapsed as *const f32).add(token*hidden+feature);squares+=value*value;feature+=32;
                }
                let mut distance=16;
                while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                if lane==0 {*(p.norm_partials as *mut f32).add(token*chunks+chunk)=squares;}
                chunk+=threads/32;
            }
            core::arch::nvptx::_syncthreads();
            if thread<32 {
                let mut squares=0.0f32;let mut chunk=lane;
                while chunk<chunks {squares+=*(p.norm_partials as *const f32).add(token*chunks+chunk);chunk+=32;}
                let mut distance=16;
                while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                store!(COEFFICIENT_CAPACITY+lane,1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs()));
            }
            core::arch::nvptx::_syncthreads();
            let inverse=load!(COEFFICIENT_CAPACITY+lane);
            let mut feature=thread;
            while feature<hidden {
                let normalized=crate::layer::round_bf16(*(p.collapsed as *const f32).add(token*hidden+feature)*inverse*1.0);
                let value=normalized * *(p.norm_weight as *const f32).add(feature);
                *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
                feature+=threads;
            }
        }
    }
}};}

macro_rules! boundary_finish_body {($arg:expr)=>{{let p=$arg;
    let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
    // CHAIN-SHADOW tail: blocks past the row grid only prefetch, then return.
    crate::chain_shadow::shadow_tail!(p,shape.rows as usize);
    if token>=shape.rows as usize {return;}
    boundary_finish_token!(p,token)
}};}

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_finish(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_finish_token. The launch binder checks the arguments.
    unsafe {boundary_finish_body!(p)}
}
/// Replay the unchanged finish body with the original grid and checked arguments.
#[inline(always)]
pub(crate) fn resident_mhc_boundary_finish(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_finish_token. The resident program checks the arguments before the replay.
    unsafe {boundary_finish_body!(p)}
}

/// One block owns `MHC_BLOCK_TOKENS` (8) consecutive tokens (256 threads, 8 warps):
/// - Phase 1 loads all `ntok * 20` coefficients once, computes all `ntok` residual rows (cache-streaming reads on
///   `previous_input`/`input`, optional packed BF16 residual read/write halving DRAM traffic and keeping residual L2-hot),
///   then all `ntok` warps compute the `ntok` RMS inverses in parallel in exact ascending column order.
/// - Phase 2 double-buffers 512-column normalized residual strips (`2 x 8 x 512 = 32 KB`, 1 barrier per 512 cols = 32
///   barriers total, overlapping next strip loads with current strip FMAs), loads all 3 weight rows back-to-back, reads
///   each token's normalized vector from shared memory once across all 3 rows, and interleaves `r = 0..3` for FMA ILP.
/// - Phase 3 runs Sinkhorn across all `ntok` warps in parallel, caches `norm_weight` in shared memory once across all
///   `ntok` tokens, and uses disjoint per-token partial slots (2 barriers/token).
#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_block(p:Boundary) {
    // SAFETY: The caller validates and retains disjoint planes (packed BF16 weights of `count*wide/2` words when bit 0
    // of `prefetch_reserved` is set; packed BF16 `residual` / `previous_input` in their first `rows*wide/2` words when
    // bits 1 / 2 are set, valid only for `streams == 4 && hidden == 4096` and 16-byte aligned pointers). Each block owns
    // disjoint tokens `base_tok..base_tok+ntok` (`base_tok = block_idx_x * MHC_BLOCK_TOKENS`, `ntok <= MHC_BLOCK_TOKENS`).
    // Every word read or written stays inside those tokens' rows, the read-only parameter planes, or the private static
    // shared array (`32,896 B`). All 256 threads execute identical token counts and column loops and reach every barrier
    // and full-mask warp shuffle together.
    unsafe {
        let shape=p.shape;
        let base_tok=core::arch::nvptx::_block_idx_x() as usize*MHC_BLOCK_TOKENS;
        if base_tok>=shape.rows as usize {return;}
        let ntok=(shape.rows as usize-base_tok).min(MHC_BLOCK_TOKENS);
        let thread=core::arch::nvptx::_thread_idx_x() as usize;let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;let warp=thread/32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        const ROWS_A_WARP:usize=3;
        if threads!=256 || count==0 || count>8*ROWS_A_WARP || wide%4!=0 || p.prefetch_reserved>7 {core::arch::asm!("trap;");}
        let bf16_weights=(p.prefetch_reserved&1)!=0;
        let out_res_bf16=(p.prefetch_reserved&2)!=0;
        let prev_res_bf16=(p.prefetch_reserved&4)!=0;
        if bf16_weights && (wide%8!=0 || (p.weights&15)!=0) {core::arch::asm!("trap;");}
        if out_res_bf16 && (streams!=4 || hidden!=4096 || (p.residual&15)!=0) {core::arch::asm!("trap;");}
        if prev_res_bf16 && (p.previous_input==0 || streams!=4 || hidden!=4096 || (p.previous_input&15)!=0) {core::arch::asm!("trap;");}
        use crate::strip_warp::{ld4,mul,add,unroll};
        // Double-buffered 512-col x 8-token normalized tile (2 * 4096 = 8192 words = 32,768 B; also stages one
        // packed BF16 residual row in Phase 1 for zero-DRAM-read fused RMS) + 160 words prev coeffs + 32 words inv
        // scratch = 33,536 B (<= 49 KB -> 2 CTAs/SM).
        const STRIP:usize=512;
        const BUF_WORDS:usize=MHC_BLOCK_TOKENS*STRIP;
        const COEFF_OFF:usize=2*BUF_WORDS;
        const INV_OFF:usize=COEFF_OFF+MHC_BLOCK_TOKENS*20;
        let smem=crate::ptx::shared::<{(INV_OFF+32)*4}>();
        macro_rules! sload {($idx:expr)=>{crate::ptx::load(smem+(($idx)*4) as u32)}}
        macro_rules! sstore {($idx:expr,$v:expr)=>{crate::ptx::store(smem+(($idx)*4) as u32,$v)}}
        macro_rules! ld4_cs {($ptr:expr)=>{{
            let (a,b,c,d):(f32,f32,f32,f32);
            core::arch::asm!("ld.global.cs.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$ptr,options(nostack,readonly));
            [a,b,c,d]
        }};}
        macro_rules! ld2_cs_bf16 {($ptr:expr)=>{{
            let (a,b):(u32,u32);
            core::arch::asm!("ld.global.cs.v2.u32 {{{a},{b}}}, [{p}];",
                a=out(reg32)a,b=out(reg32)b,p=in(reg64)$ptr,options(nostack,readonly));
            [f32::from_bits(a<<16),f32::from_bits(a&0xffff_0000),f32::from_bits(b<<16),f32::from_bits(b&0xffff_0000)]
        }};}
        macro_rules! st2_bf16 {($ptr:expr,$o0:expr,$o1:expr,$o2:expr,$o3:expr)=>{{
            let w0=($o0.to_bits()>>16)|($o1.to_bits()&0xffff_0000);
            let w1=($o2.to_bits()>>16)|($o3.to_bits()&0xffff_0000);
            core::arch::asm!("st.global.v2.u32 [{at}], {{{a},{b}}};",
                a=in(reg32)w0,b=in(reg32)w1,at=in(reg64)$ptr,options(nostack));
        }};}
        // Phase 1: load all `ntok * 20` previous coefficients once into `COEFF_OFF`, then produce all `ntok` residual
        // rows. Phase 1.5 runs all `ntok` warps in parallel over the L2-hot residual tile (1 barrier total).
        if p.previous_input!=0 && streams==4 {
            if thread<ntok*20 {
                let tok_i=thread/20;let ci=thread%20;
                let coeffs=(p.previous_coefficients as *const f32).add((base_tok+tok_i)*count);
                let v=if ci<16 {crate::bf16::rounded(*coeffs.add(8+ci))} else {crate::bf16::rounded(*coeffs.add(4+(ci-16)))};
                sstore!(COEFF_OFF+thread,v);
            }
            core::arch::nvptx::_syncthreads();
        }
        let mut t=0usize;
        while t<ntok {
            let token=base_tok+t;
            let residual=(p.residual as *mut f32).add(token*wide);
            if p.previous_input!=0 && streams==4 {
                let input=(p.input as *const f32).add(token*hidden);
                let cbase=COEFF_OFF+t*20;
                if hidden%4==0 && ((p.previous_input|p.input|p.residual)&15)==0 {
                    let prev_f32=(p.previous_input as *const f32).add(token*wide);
                    let prev_u32=(p.previous_input as *const u32).add(token*(wide/2));
                    let res_u32=(p.residual as *mut u32).add(token*(wide/2));
                    let h2=hidden/2;
                    let mut feature=thread*4;
                    while feature<hidden {
                        let (p0,p1,p2,p3)=if prev_res_bf16 {
                            let f2=feature/2;
                            (ld2_cs_bf16!(prev_u32.add(f2)),ld2_cs_bf16!(prev_u32.add(h2+f2)),
                             ld2_cs_bf16!(prev_u32.add(2*h2+f2)),ld2_cs_bf16!(prev_u32.add(3*h2+f2)))
                        } else {
                            (ld4_cs!(prev_f32.add(feature)),ld4_cs!(prev_f32.add(hidden+feature)),
                             ld4_cs!(prev_f32.add(2*hidden+feature)),ld4_cs!(prev_f32.add(3*hidden+feature)))
                        };
                        let blk=ld4_cs!(input.add(feature));
                        let b0=crate::bf16::rounded(blk[0]);let b1=crate::bf16::rounded(blk[1]);
                        let b2=crate::bf16::rounded(blk[2]);let b3=crate::bf16::rounded(blk[3]);
                        let mut dst=0usize;
                        while dst<4 {
                            let c0=sload!(cbase+dst);let c1=sload!(cbase+4+dst);let c2=sload!(cbase+8+dst);let c3=sload!(cbase+12+dst);
                            let pd=sload!(cbase+16+dst);
                            let mut s0=0.0f32;add!(s0,mul!(c0,p0[0]));add!(s0,mul!(c1,p1[0]));add!(s0,mul!(c2,p2[0]));add!(s0,mul!(c3,p3[0]));
                            let mut s1=0.0f32;add!(s1,mul!(c0,p0[1]));add!(s1,mul!(c1,p1[1]));add!(s1,mul!(c2,p2[1]));add!(s1,mul!(c3,p3[1]));
                            let mut s2=0.0f32;add!(s2,mul!(c0,p0[2]));add!(s2,mul!(c1,p1[2]));add!(s2,mul!(c2,p2[2]));add!(s2,mul!(c3,p3[2]));
                            let mut s3=0.0f32;add!(s3,mul!(c0,p0[3]));add!(s3,mul!(c1,p1[3]));add!(s3,mul!(c2,p2[3]));add!(s3,mul!(c3,p3[3]));
                            let o0=crate::bf16::rounded(crate::bf16::rounded(s0)+crate::bf16::rounded(mul!(pd,b0)));
                            let o1=crate::bf16::rounded(crate::bf16::rounded(s1)+crate::bf16::rounded(mul!(pd,b1)));
                            let o2=crate::bf16::rounded(crate::bf16::rounded(s2)+crate::bf16::rounded(mul!(pd,b2)));
                            let o3=crate::bf16::rounded(crate::bf16::rounded(s3)+crate::bf16::rounded(mul!(pd,b3)));
                            if out_res_bf16 {
                                st2_bf16!(res_u32.add(dst*h2+feature/2),o0,o1,o2,o3);
                            } else {
                                core::arch::asm!("st.global.v4.f32 [{at}], {{{a},{b},{c},{d}}};",
                                    a=in(reg32)o0,b=in(reg32)o1,c=in(reg32)o2,d=in(reg32)o3,
                                    at=in(reg64)residual.add(dst*hidden+feature),options(nostack));
                            }
                            dst+=1;
                        }
                        feature+=threads*4;
                    }
                } else {
                    let previous=(p.previous_input as *const f32).add(token*wide);
                    let mut feature=thread;
                    while feature<hidden {
                        let v0=*previous.add(feature);let v1=*previous.add(hidden+feature);
                        let v2=*previous.add(2*hidden+feature);let v3=*previous.add(3*hidden+feature);
                        let block=crate::bf16::rounded(*input.add(feature));
                        unroll!(4,dst,{
                            let mut sum=0.0f32;
                            add!(sum,mul!(sload!(cbase+dst),v0));add!(sum,mul!(sload!(cbase+4+dst),v1));
                            add!(sum,mul!(sload!(cbase+8+dst),v2));add!(sum,mul!(sload!(cbase+12+dst),v3));
                            let product=crate::bf16::rounded(mul!(sload!(cbase+16+dst),block));
                            *residual.add(dst*hidden+feature)=crate::bf16::rounded(crate::bf16::rounded(sum)+product);
                        });
                        feature+=threads;
                    }
                }
            } else if p.previous_input==0 && out_res_bf16 {
                let res_u32=(p.residual as *mut u32).add(token*(wide/2));
                if (p.input&15)==0 {
                    let inp=(p.input as *const f32).add(token*wide);
                    let mut feature=thread*4;
                    while feature<wide {
                        let x=ld4_cs!(inp.add(feature));
                        let o0=crate::bf16::rounded(x[0]);let o1=crate::bf16::rounded(x[1]);
                        let o2=crate::bf16::rounded(x[2]);let o3=crate::bf16::rounded(x[3]);
                        st2_bf16!(res_u32.add(feature/2),o0,o1,o2,o3);
                        feature+=threads*4;
                    }
                } else {
                    let inp=(p.input as *const f32).add(token*wide);
                    let mut pair=thread;
                    while pair<wide/2 {
                        let o0=crate::bf16::rounded(*inp.add(pair*2));
                        let o1=crate::bf16::rounded(*inp.add(pair*2+1));
                        *res_u32.add(pair)=(o0.to_bits()>>16)|(o1.to_bits()&0xffff_0000);
                        pair+=threads;
                    }
                }
            } else {
                let mut feature=thread;
                while feature<wide {
                    let value=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                        else {crate::mhc_boundary_math::combine(
                            slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                            slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                            crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature%hidden)),
                            streams,hidden,feature/hidden,feature%hidden)};
                    *residual.add(feature)=value;feature+=threads;
                }
            }
            t+=1;
        }
        core::arch::nvptx::_syncthreads();
        // Phase 1.5: all `ntok` warps compute exact RMS inverses in parallel from the L2-hot residual tile.
        if warp<ntok {
            let mut squares=0.0f32;let mut col=lane*8;
            if out_res_bf16 {
                let res_u32=(p.residual as *const u32).add((base_tok+warp)*(wide/2));
                while col<wide {
                    let (p0,p1,p2,p3):(u32,u32,u32,u32);
                    core::arch::asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",
                        a=out(reg32)p0,b=out(reg32)p1,c=out(reg32)p2,d=out(reg32)p3,
                        p=in(reg64)res_u32.add(col/2),options(nostack,readonly));
                    let ps=[p0,p1,p2,p3];
                    unroll!(4,q,{
                        let a=f32::from_bits(ps[q]<<16);let b=f32::from_bits(ps[q]&0xffff_0000);
                        add!(squares,mul!(a,a));add!(squares,mul!(b,b));
                    });
                    col+=256;
                }
            } else {
                let res_ro=(p.residual as *const f32).add((base_tok+warp)*wide);
                while col<wide {
                    let halves=if col+4<wide {2} else {1};
                    unroll!(2,h,{if h<halves {let x=ld4!(res_ro.add(col+4*h));unroll!(4,e,{add!(squares,mul!(x[e],x[e]));});}});
                    col+=256;
                }
            }
            let inv=1.0/crate::qsa_common::root(crate::reduction::warp_root(squares)/wide as f32+shape.rms_eps);
            if lane==0 {sstore!(INV_OFF+warp,inv);}
        }
        core::arch::nvptx::_syncthreads();
        // Phase 2: double-buffered 512-column normalized strips (1 barrier per 512 cols = 32 barriers total),
        // overlapping next strip's packed BF16 loads with current strip's 3-row interleaved FMAs.
        let mut acc=[[0.0f32;MHC_BLOCK_TOKENS];ROWS_A_WARP];
        let load_tk=warp;
        let load_off=lane*16;
        let my_inv=if load_tk<ntok {sload!(INV_OFF+load_tk)} else {0.0f32};
        macro_rules! stage_bf16_strip {($buf:expr,$strip_col0:expr)=>{{
            if load_tk<ntok {
                let base_ptr=(p.residual as *const u32).add((base_tok+load_tk)*(wide/2)+($strip_col0+load_off)/2);
                let dst_base=$buf+((load_tk*STRIP+load_off)*4) as u32;
                unroll!(2,q,{
                    let (p0,p1,p2,p3):(u32,u32,u32,u32);
                    core::arch::asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",
                        a=out(reg32)p0,b=out(reg32)p1,c=out(reg32)p2,d=out(reg32)p3,
                        p=in(reg64)base_ptr.add(q*4),options(nostack,readonly));
                    let n0=mul!(f32::from_bits(p0<<16),my_inv);let n1=mul!(f32::from_bits(p0&0xffff_0000),my_inv);
                    let n2=mul!(f32::from_bits(p1<<16),my_inv);let n3=mul!(f32::from_bits(p1&0xffff_0000),my_inv);
                    core::arch::asm!("st.shared.v4.f32 [{at}], {{{a},{b},{c},{d}}};",
                        a=in(reg32)n0,b=in(reg32)n1,c=in(reg32)n2,d=in(reg32)n3,at=in(reg32)dst_base+(q*32) as u32,options(nostack));
                    let n4=mul!(f32::from_bits(p2<<16),my_inv);let n5=mul!(f32::from_bits(p2&0xffff_0000),my_inv);
                    let n6=mul!(f32::from_bits(p3<<16),my_inv);let n7=mul!(f32::from_bits(p3&0xffff_0000),my_inv);
                    core::arch::asm!("st.shared.v4.f32 [{at}], {{{a},{b},{c},{d}}};",
                        a=in(reg32)n4,b=in(reg32)n5,c=in(reg32)n6,d=in(reg32)n7,at=in(reg32)dst_base+(q*32+16) as u32,options(nostack));
                });
            }
        }};}
        macro_rules! stage_f32_strip {($buf:expr,$strip_col0:expr)=>{{
            if load_tk<ntok {
                let res_tok=(p.residual as *const f32).add((base_tok+load_tk)*wide);
                let norm_tok=(p.mixing_normalized as *mut f32).wrapping_add((base_tok+load_tk)*wide);
                unroll!(4,q,{
                    let c_at=$strip_col0+load_off+q*4;
                    if c_at<wide {
                        let x=ld4!(res_tok.add(c_at));
                        let n0=mul!(x[0],my_inv);let n1=mul!(x[1],my_inv);let n2=mul!(x[2],my_inv);let n3=mul!(x[3],my_inv);
                        if p.mixing_normalized!=0 {
                            core::arch::asm!("st.global.v4.f32 [{at}], {{{a},{b},{c},{d}}};",
                                a=in(reg32)n0,b=in(reg32)n1,c=in(reg32)n2,d=in(reg32)n3,
                                at=in(reg64)norm_tok.wrapping_add(c_at),options(nostack));
                        }
                        let sdst=$buf+((load_tk*STRIP+load_off+q*4)*4) as u32;
                        core::arch::asm!("st.shared.v4.f32 [{at}], {{{a},{b},{c},{d}}};",
                            a=in(reg32)n0,b=in(reg32)n1,c=in(reg32)n2,d=in(reg32)n3,at=in(reg32)sdst,options(nostack));
                    }
                });
            }
        }};}
        let total_iters=wide.div_ceil(STRIP);
        if out_res_bf16 {stage_bf16_strip!(smem,0usize);} else {stage_f32_strip!(smem,0usize);}
        core::arch::nvptx::_syncthreads();
        let mut iter=0usize;
        while iter<total_iters {
            let cur_buf=smem+((iter&1)*BUF_WORDS*4) as u32;
            let next_buf=smem+(((iter+1)&1)*BUF_WORDS*4) as u32;
            let col0=iter*STRIP;
            if iter+1<total_iters {
                if out_res_bf16 {stage_bf16_strip!(next_buf,col0+STRIP);} else {stage_f32_strip!(next_buf,col0+STRIP);}
            }
            let mut half=0usize;
            while half<2 {
                let mut cb_l=cur_buf;
                core::arch::asm!("/* {0} */",inout(reg32)cb_l,options(nomem,nostack));
                let col=col0+half*256+lane*8;
                if col<wide {
                    if bf16_weights {
                        let weights=p.weights as *const u32;
                        // Four 2-column chunks (`ld.global.nc.u32` = 3 live weight regs + `ld.shared.v2.u32` = 2 regs -> 0 spills under 128-reg cap).
                        let mut ch=0usize;
                        while ch<4 {
                            let mut pw=[0u32;3];
                            unroll!(3,r,{
                                let row_idx=(warp*ROWS_A_WARP+r).min(count-1);
                                let w:u32;
                                core::arch::asm!("ld.global.nc.u32 {w}, [{p}];",
                                    w=out(reg32)w,p=in(reg64)weights.add(row_idx*(wide/2)+col/2+ch),options(nostack,readonly));
                                pw[r]=w;
                            });
                            unroll!(8,t,{if t<ntok {
                                let nv=crate::ptx::load2(cb_l+((t*STRIP+half*256+lane*8+ch*2)*4) as u32);
                                let f0=f32::from_bits(nv[0]);let f1=f32::from_bits(nv[1]);
                                unroll!(3,r,{add!(acc[r][t],mul!(f32::from_bits(pw[r]<<16),f0));});
                                unroll!(3,r,{add!(acc[r][t],mul!(f32::from_bits(pw[r]&0xffff_0000),f1));});
                            }});
                            ch+=1;
                        }
                    } else {
                        let weights=p.weights as *const f32;
                        let halves=if col+4<wide {2} else {1};
                        let mut h=0usize;
                        while h<halves {
                            unroll!(3,r,{
                                let rowp=weights.add((warp*ROWS_A_WARP+r).min(count-1)*wide);
                                let w=ld4!(rowp.add(col+4*h));
                                unroll!(8,t,{if t<ntok {
                                    let n=crate::ptx::load4(cb_l+((t*STRIP+half*256+lane*8+4*h)*4) as u32);
                                    unroll!(4,e,{add!(acc[r][t],mul!(w[e],f32::from_bits(n[e])));});
                                }});
                            });
                            h+=1;
                        }
                    }
                }
                half+=1;
            }
            core::arch::nvptx::_syncthreads();
            iter+=1;
        }
        unroll!(8,t,{if t<ntok {
            let token=base_tok+t;
            unroll!(3,r,{
                let coefficient=warp*ROWS_A_WARP+r;
                if coefficient<count {
                    let total=crate::reduction::warp_root(acc[r][t]);
                    if lane==0 {
                        let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                        *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
                    }
                }
            });
        }});
        core::arch::nvptx::_syncthreads();
        // Phase 3: fast parallel 8-warp Sinkhorn + cached norm_weight + disjoint per-token partials for the 4-stream shape.
        let chunks=hidden.div_ceil(128);
        if streams==4 && hidden%128==0 && chunks<=32 {
            // Parallel 8-warp Sinkhorn: warp `t < ntok` activates token `base_tok + t`'s coefficients.
            if warp<ntok {
                let token=base_tok+warp;
                let coefficients=(p.coefficients as *mut f32).add(token*count);
                let pre=if lane<4 {math::sigmoid(*coefficients.add(lane))+shape.eps} else {0.0};
                let post=if lane<4 {2.0*math::sigmoid(*coefficients.add(4+lane))} else {0.0};
                let mut value=if lane<16 {*coefficients.add(8+lane)} else {0.0};
                let row=(lane/4).min(3);let column=lane%4;
                macro_rules! read_mask {($mask:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),$mask,31))}}
                let along=[(column^0) as u32,(column^1) as u32,(column^2) as u32,(column^3) as u32];
                let down=[((row^0)*4) as u32,((row^1)*4) as u32,((row^2)*4) as u32,((row^3)*4) as u32];
                let mut maximum=f32::NEG_INFINITY;
                unroll!(4,source,{maximum=maximum.max(read_mask!(along[source]));});
                value=crate::qsa_common::exponential(value-maximum);
                let mut sum=0.0f32;
                unroll!(4,source,{sum+=read_mask!(along[source]);});
                value=value/sum+shape.eps;
                let mut sum=0.0f32;
                unroll!(4,source,{sum+=read_mask!(down[source]);});
                value/=sum+shape.eps;
                for _iteration in 1..shape.iterations {
                    let mut sum=0.0f32;unroll!(4,source,{sum+=read_mask!(along[source]);});
                    value/=sum+shape.eps;
                    let mut sum=0.0f32;unroll!(4,source,{sum+=read_mask!(down[source]);});
                    value/=sum+shape.eps;
                }
                if lane<4 {*coefficients.add(lane)=pre;*coefficients.add(4+lane)=post;sstore!(warp*4+lane,pre);}
                if lane<16 {*coefficients.add(8+lane)=value;}
            }
            // Cooperatively cache norm_weight[0..hidden] into smem[512..512+hidden] once across all `ntok` tokens.
            const NW_OFF:usize=512;
            {
                let nw_ptr=p.norm_weight as *const f32;
                let mut f=thread*4;
                while f<hidden {
                    let w=ld4!(nw_ptr.add(f));
                    let sdst=smem+((NW_OFF+f)*4) as u32;
                    core::arch::asm!("st.shared.v4.f32 [{at}], {{{a},{b},{c},{d}}};",
                        a=in(reg32)w[0],b=in(reg32)w[1],c=in(reg32)w[2],d=in(reg32)w[3],at=in(reg32)sdst,options(nostack));
                    f+=threads*4;
                }
            }
            core::arch::nvptx::_syncthreads();
            // Pack each thread's 16 already-BF16-rounded held values into 8 u32 words in free `smem[HELD_OFF..]`
            // (`4608 + 256*8 = 6656 <= 8192` words) to free 16 live registers (0 spills under 128-reg cap).
            const HELD_OFF:usize=NW_OFF+4096;
            let my_held=smem+((HELD_OFF+thread*8)*4) as u32;
            let h2=hidden/2;
            let odd_lane=(lane&1)!=0;
            let mut t=0usize;
            while t<ntok {
                let token=base_tok+t;
                let part_off=32+t*32;
                let pre0=sload!(t*4);let pre1=sload!(t*4+1);let pre2=sload!(t*4+2);let pre3=sload!(t*4+3);
                let res_tok=(p.residual as *const f32).add(token*wide);
                let res_tok_u32=(p.residual as *const u32).add(token*(wide/2));
                let mut wave=0usize;
                while wave<4 {
                    let chunk=warp+wave*8;
                    if chunk<chunks {
                        let off=chunk*128+lane;
                        let mut value=[0.0f32;4];
                        if out_res_bf16 {
                            let off2=off/2;
                            macro_rules! ld_u32_nc {($ptr:expr)=>{{let v:u32;core::arch::asm!("ld.global.nc.u32 {v}, [{p}];",v=out(reg32)v,p=in(reg64)$ptr,options(nostack,readonly));v}};}
                            unroll!(4,e,{
                                let w=ld_u32_nc!(res_tok_u32.add(off2+16*e));
                                let x=f32::from_bits(if odd_lane {w&0xffff_0000} else {w<<16});
                                add!(value[e],mul!(pre0,x));
                            });
                            unroll!(4,e,{
                                let w=ld_u32_nc!(res_tok_u32.add(h2+off2+16*e));
                                let x=f32::from_bits(if odd_lane {w&0xffff_0000} else {w<<16});
                                add!(value[e],mul!(pre1,x));
                            });
                            unroll!(4,e,{
                                let w=ld_u32_nc!(res_tok_u32.add(2*h2+off2+16*e));
                                let x=f32::from_bits(if odd_lane {w&0xffff_0000} else {w<<16});
                                add!(value[e],mul!(pre2,x));
                            });
                            unroll!(4,e,{
                                let w=ld_u32_nc!(res_tok_u32.add(3*h2+off2+16*e));
                                let x=f32::from_bits(if odd_lane {w&0xffff_0000} else {w<<16});
                                add!(value[e],mul!(pre3,x));
                            });
                        } else {
                            unroll!(4,e,{let x=*res_tok.add(off+32*e);add!(value[e],mul!(pre0,x));});
                            unroll!(4,e,{let x=*res_tok.add(hidden+off+32*e);add!(value[e],mul!(pre1,x));});
                            unroll!(4,e,{let x=*res_tok.add(2*hidden+off+32*e);add!(value[e],mul!(pre2,x));});
                            unroll!(4,e,{let x=*res_tok.add(3*hidden+off+32*e);add!(value[e],mul!(pre3,x));});
                        }
                        let mut squares=0.0f32;
                        let v0=crate::bf16::rounded(value[0]);let v1=crate::bf16::rounded(value[1]);
                        let v2=crate::bf16::rounded(value[2]);let v3=crate::bf16::rounded(value[3]);
                        if p.collapsed!=0 {
                            *(p.collapsed as *mut f32).add(token*hidden+off)=v0;
                            *(p.collapsed as *mut f32).add(token*hidden+off+32)=v1;
                            *(p.collapsed as *mut f32).add(token*hidden+off+64)=v2;
                            *(p.collapsed as *mut f32).add(token*hidden+off+96)=v3;
                        }
                        add!(squares,mul!(v0,v0));add!(squares,mul!(v1,v1));add!(squares,mul!(v2,v2));add!(squares,mul!(v3,v3));
                        let pk0=(v0.to_bits()>>16)|(v1.to_bits()&0xffff_0000);
                        let pk1=(v2.to_bits()>>16)|(v3.to_bits()&0xffff_0000);
                        core::arch::asm!("st.shared.v2.u32 [{p}], {{{a},{b}}};",
                            a=in(reg32)pk0,b=in(reg32)pk1,p=in(reg32)my_held+(wave*8) as u32,options(nostack));
                        let mut distance=16;
                        while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                        if lane==0 {
                            if p.norm_partials!=0 {*(p.norm_partials as *mut f32).add(token*chunks+chunk)=squares;}
                            else {sstore!(part_off+chunk,squares);}
                        }
                    }
                    wave+=1;
                }
                core::arch::nvptx::_syncthreads();
                if warp==0 {
                    let mut squares=0.0f32;let mut chunk=lane;
                    while chunk<chunks {
                        squares+=if p.norm_partials!=0 {*(p.norm_partials as *const f32).add(token*chunks+chunk)} else {sload!(part_off+chunk)};
                        chunk+=32;
                    }
                    let mut distance=16;
                    while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                    sstore!(part_off+lane,1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs()));
                }
                core::arch::nvptx::_syncthreads();
                let inverse=sload!(part_off+lane);
                let mut wave=0usize;
                while wave<4 {
                    let chunk=warp+wave*8;
                    if chunk<chunks {
                        let (pk0,pk1):(u32,u32);
                        core::arch::asm!("ld.shared.v2.u32 {{{a},{b}}}, [{p}];",
                            a=out(reg32)pk0,b=out(reg32)pk1,p=in(reg32)my_held+(wave*8) as u32,options(nostack,readonly));
                        let hv=[f32::from_bits(pk0<<16),f32::from_bits(pk0&0xffff_0000),f32::from_bits(pk1<<16),f32::from_bits(pk1&0xffff_0000)];
                        unroll!(4,e,{let feature=chunk*128+lane+32*e;
                            let normalized=crate::layer::round_bf16(hv[e]*inverse*1.0);
                            let value=normalized * sload!(NW_OFF+feature);
                            *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
                        });
                    }
                    wave+=1;
                }
                t+=1;
            }
        } else {
            // Low-register scalar fallback reusing `smem` (no second shared allocation or 16-reg unrolled arrays).
            let mut t=0usize;
            while t<ntok {
                let token=base_tok+t;
                if thread==0 {math::activate(shape,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(token*count),count));}
                core::arch::nvptx::_syncthreads();
                let mut feature=thread;
                while feature<hidden {
                    *(p.collapsed as *mut f32).add(token*hidden+feature)=crate::mhc_boundary_math::collapse(
                        slice::from_raw_parts((p.residual as *const f32).add(token*wide),wide),
                        slice::from_raw_parts((p.coefficients as *const f32).add(token*count),streams),streams,hidden,feature);
                    feature+=threads;
                }
                core::arch::nvptx::_syncthreads();
                let mut chunk=warp;
                while chunk<chunks {
                    let mut squares=0.0f32;let mut feature=chunk*128+lane;
                    while feature<(chunk*128+128).min(hidden) {
                        let value=*(p.collapsed as *const f32).add(token*hidden+feature);squares+=value*value;feature+=32;
                    }
                    let mut distance=16;
                    while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                    if lane==0 {*(p.norm_partials as *mut f32).add(token*chunks+chunk)=squares;}
                    chunk+=8;
                }
                core::arch::nvptx::_syncthreads();
                if warp==0 {
                    let mut squares=0.0f32;let mut chunk=lane;
                    while chunk<chunks {squares+=*(p.norm_partials as *const f32).add(token*chunks+chunk);chunk+=32;}
                    let mut distance=16;
                    while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                    sstore!(INV_OFF+lane,1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs()));
                }
                core::arch::nvptx::_syncthreads();
                let inverse=sload!(INV_OFF+lane);
                let mut feature=thread;
                while feature<hidden {
                    let normalized=crate::layer::round_bf16(*(p.collapsed as *const f32).add(token*hidden+feature)*inverse*1.0);
                    let value=normalized * *(p.norm_weight as *const f32).add(feature);
                    *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
                    feature+=threads;
                }
                core::arch::nvptx::_syncthreads();
                t+=1;
            }
        }
    }
}
