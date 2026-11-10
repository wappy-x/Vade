//! `mhc_boundary_project` for a group of tokens: the same planes, the same arithmetic, the same ascending FP32 chains,
//! no fused multiply-add. The one-block project gives each token a block of 768 threads (one block per SM), so a
//! 256-row prompt group is five waves, and its staged pipeline (a barrier and a `cp.async` wait per 128 features) costs
//! 1.3 us a step. Here a block of TOKENS warps owns TOKENS tokens and has no barrier at all (the residual plane comes from `mhc_residual_rows`, an earlier launch):
//! warp t is the chain warp of token t. Lane 0 folds the RMS chain, then every lane r < coefficients folds coefficient
//! row r over the normalized residual, reading the residual and the weight row straight from global memory with
//! L1 prefetch twenty lines ahead; the warps of a block read the same weight lines at about the same time, so the
//! weights come from L2 about once per block. Every chain still starts at 0.0 and takes the rounded products in feature
//! order.
//!
//! Launch: at least `ceil(rows / TOKENS)` blocks of TOKENS * 32 threads; shapes by `prompt_admission::mhc_tile_admitted`.
use crate::mhc_params::Boundary;
use core::slice;

pub(crate) const TOKENS:usize=crate::prompt_admission::MHC_TOKENS;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_project_tile(p:Boundary) {
    // SAFETY: The caller (backend/mhc_boundary.rs, after Shape::validate and the plane checks of mhc_planes) supplies
    // validated, live, disjoint planes that outlive the ordered consumer, and launches blocks of exactly TOKENS*32 threads
    // for a shape with mhc_tile_admitted (coefficients <= 24, wide divisible by 4); any other block size or shape traps.
    // Block b owns tokens [b*TOKENS, min(rows, (b+1)*TOKENS)); blocks past the last token return before any access. Every
    // global write is to a word of one of those tokens' residual / normalized / coefficient rows, by one thread each
    // (normalized plane: a lane of the token's warp; coefficient: lane r of the token's warp). There is no barrier, so the early
    // returns are free. Every read is of an input plane or of the residual plane, complete since the earlier launch on this stream;
    // all loads are inside those rows (16-byte loads at multiples of four words, wide is divisible by 4). A prefetch address
    // is only built when it is inside its row (prompt_admission::prefetch_inside: i + AHEAD < wide, tested at the boundary),
    // so no pointer is ever advanced past an allocation. No shared memory.
    unsafe {
        let shape=p.shape;let rows=shape.rows as usize;
        let first=core::arch::nvptx::_block_idx_x() as usize*TOKENS;
        if first>=rows {return;}
        let tb=(rows-first).min(TOKENS);
        let thread=core::arch::nvptx::_thread_idx_x() as usize;let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;let warp=thread/32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;
        if threads!=TOKENS*32 || count>24 || wide%4!=0 {core::arch::asm!("trap;");}
        // The residual plane was written by mhc_residual_rows, an earlier launch on the same stream.
        if warp>=tb {return;}
        let token=first+warp;
        let residual=(p.residual as *const f32).add(token*wide);let weights=p.weights as *const f32;
        macro_rules! load4 {($at:expr)=>{{
            let (a,b,c,d):(u32,u32,u32,u32);
            core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{at}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,at=in(reg64)$at,options(nostack,readonly));
            [f32::from_bits(a),f32::from_bits(b),f32::from_bits(c),f32::from_bits(d)]
        }};}
        macro_rules! prefetch {($at:expr)=>{core::arch::asm!("prefetch.global.L1 [{a}];",a=in(reg64)$at,options(nostack))};}
        const AHEAD:usize=640;
        // The RMS sum: lane 0 folds the squares in ascending feature order.
        let mut inverse=0.0f32;
        if lane==0 {
            let mut squares=0.0f32;let mut i=0usize;
            while i+16<=wide {
                if crate::prompt_admission::prefetch_inside(i,AHEAD,wide) {prefetch!(residual.add(i+AHEAD));}
                let v=[load4!(residual.add(i)),load4!(residual.add(i+4)),load4!(residual.add(i+8)),load4!(residual.add(i+12))];
                for q in 0..4 {for e in 0..4 {let x=v[q][e];let square=x*x;squares+=square;}}
                i+=16;
            }
            while i<wide {let v=load4!(residual.add(i));for e in 0..4 {let x=v[e];let square=x*x;squares+=square;}i+=4;}
            inverse=1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps);
        }
        let inverse:f32={let o:u32;core::arch::asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)o,v=in(reg32)inverse.to_bits(),options(nomem,nostack));f32::from_bits(o)};
        // The normalized plane of this token, coalesced over the lanes.
        let mut f=lane;
        while f<wide {*(p.mixing_normalized as *mut f32).add(token*wide+f)=*residual.add(f)*inverse;f+=32;}
        // The dots: lane r folds coefficient row r.
        if lane<count {
            let row=weights.add(lane*wide);let mut acc=0.0f32;let mut i=0usize;
            while i+16<=wide {
                if crate::prompt_admission::prefetch_inside(i,AHEAD,wide) {prefetch!(row.add(i+AHEAD));prefetch!(residual.add(i+AHEAD));}
                let w=[load4!(row.add(i)),load4!(row.add(i+4)),load4!(row.add(i+8)),load4!(row.add(i+12))];
                let x=[load4!(residual.add(i)),load4!(residual.add(i+4)),load4!(residual.add(i+8)),load4!(residual.add(i+12))];
                for q in 0..4 {for e in 0..4 {let normalized=x[q][e]*inverse;let product=w[q][e]*normalized;acc+=product;}}
                i+=16;
            }
            while i<wide {
                let w=load4!(row.add(i));let x=load4!(residual.add(i));
                for e in 0..4 {let normalized=x[e]*inverse;let product=w[e]*normalized;acc+=product;}
                i+=4;
            }
            let group=if lane<streams {0} else if lane<2*streams {1} else {2};
            *(p.coefficients as *mut f32).add(token*count+lane)=acc * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(lane);
        }
    }
}

/// The residual plane, one thread per feature: `mhc_boundary_project`'s first loop, as its own launch so that every
/// feature of the group has a thread (the loop inside a tile block would give each thread hundreds of dependent loads).
#[no_mangle]
pub extern "ptx-kernel" fn mhc_residual_rows(p:Boundary) {
    // SAFETY: As mhc_project_tile for the planes (validated, live, disjoint; previous planes read only when present, with the
    // same token and stream widths). Thread i owns the residual word i of the group (i < rows*wide, the guard), so every
    // word has one writer; all reads are inside the token's input, previous residual and previous coefficient rows.
    unsafe {
        let shape=p.shape;let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let i=crate::index();if i>=shape.rows as usize*wide {return;}
        let token=i/wide;let f=i%wide;
        let value=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+f))}
            else {crate::mhc_boundary_math::combine(
                slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+f%hidden)),
                streams,hidden,f/hidden,f%hidden)};
        *(p.residual as *mut f32).add(i)=value;
    }
}
