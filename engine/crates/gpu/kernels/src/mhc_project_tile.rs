//! Token-group mHC projection with the same fixed feature tree as decode.
use crate::mhc_params::Boundary;
use core::slice;

pub(crate) const TOKENS:usize=crate::prompt_admission::MHC_TOKENS;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_project_tile(p:Boundary) {
    // SAFETY: The caller validates and retains disjoint planes. Each complete
    // warp owns one token. Warps past the last token return together. The prior
    // residual launch completes before any read here. All lanes reach every
    // full-mask shuffle. Distinct lanes write distinct normalized words; lane
    // zero alone writes that token's coefficients. All loads stay in checked rows.
    unsafe {
        let shape=p.shape;let rows=shape.rows as usize;
        let first=core::arch::nvptx::_block_idx_x() as usize*TOKENS;
        if first>=rows {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;let warp=thread/32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;
        if threads!=TOKENS*32 || count>24 || wide%4!=0 {core::arch::asm!("trap;");}
        if warp>=rows-first {return;}
        let token=first+warp;let residual=(p.residual as *const f32).add(token*wide);
        // Lane = strip: lane L reads the 8-feature groups L, L + 32, ... of the token's rows with 16-byte loads (wide % 4 == 0,
        // so a group's second half exists only when it starts below wide) and adds in ascending features from +0.0;
        // warp_root joins the 32 strips, the fixed feature tree. Every lane holds every root.
        macro_rules! ld4 {($at:expr)=>{{
            let (a,b,c,d):(u32,u32,u32,u32);
            core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{at}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,at=in(reg64)$at,options(pure,readonly,nostack));
            [f32::from_bits(a),f32::from_bits(b),f32::from_bits(c),f32::from_bits(d)]
        }};}
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;core::arch::asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(pure,nomem,nostack));value}};}
        macro_rules! add {($sum:expr,$value:expr)=>{{let value:f32=$value;core::arch::asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(pure,nomem,nostack));}};}
        use crate::strip_warp::unroll;
        let mut squares=0.0f32;let mut col=lane*8;
        while col<wide {
            let halves=if col+4<wide {2} else {1};
            unroll!(2,h,{if h<halves {let x=ld4!(residual.add(col+4*h));unroll!(4,e,{add!(squares,mul!(x[e],x[e]));});}});
            col+=256;
        }
        let inverse=1.0/crate::qsa_common::root(crate::reduction::warp_root(squares)/wide as f32+shape.rms_eps);
        let mut feature=lane;
        while feature<wide {*(p.mixing_normalized as *mut f32).add(token*wide+feature)=*residual.add(feature)*inverse;feature+=32;}
        // Eight coefficient rows a pass: the normalized group is made once (residual * inverse, as the normalized plane)
        // and multiplied by each row's weights (weight * normalized), one add per product in the row's strip.
        const RB:usize=8;
        let weights=p.weights as *const f32;let mut c0=0;
        while c0<count {
            let mut acc=[0.0f32;RB];let mut rowp=[weights;RB];
            unroll!(8,r,{rowp[r]=weights.add((c0+r).min(count-1)*wide);});
            let mut col=lane*8;
            while col<wide {
                let halves=if col+4<wide {2} else {1};
                unroll!(2,h,{if h<halves {
                    let x=ld4!(residual.add(col+4*h));let n=[mul!(x[0],inverse),mul!(x[1],inverse),mul!(x[2],inverse),mul!(x[3],inverse)];
                    let mut w=[[0.0f32;4];RB];
                    unroll!(8,r,{w[r]=ld4!(rowp[r].add(col+4*h));});
                    unroll!(8,r,{unroll!(4,e,{add!(acc[r],mul!(w[r][e],n[e]));});});
                }});
                col+=256;
            }
            unroll!(8,r,{
                let coefficient=c0+r;
                if coefficient<count {
                    let total=crate::reduction::warp_root(acc[r]);
                    if lane==0 {
                        let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                        *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
                    }
                }
            });
            c0+=RB;
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
