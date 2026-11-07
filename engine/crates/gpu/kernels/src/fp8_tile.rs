//! Block-FP8 decode projection, shared-memory tile body. Same arithmetic and order as
//! `serial_projection::project!` for `Op::Fp8Serial`: every output row is ONE ascending FP32 chain over
//! its columns from +0.0, each product `(e4m3(w) * scale) * bf16_round(x)` made by two separate
//! `mul.rn.f32`, each step one separate `add.rn.f32`, no fused multiply-add, and the same final BF16 round.
//! What changes is who does the work. The old body gave each row to one thread that waited for each of its
//! own loads in turn (one warp per 32 rows: 32 of 48 SMs busy, one load in flight, 30 GB/s). Here a block
//! owns R rows. R*QN producer threads copy the weights with asynchronous copies (D stages deep), decode them
//! and make the products into shared memory; one warp of R chain lanes (lane l owns row l) adds the products
//! of its row in column order while the producers make the next chunk of QN*16 columns.
//!
//! Launch: the same `LaunchArguments` and job table as `expert_operation`; blocks of 32 + R*QN threads,
//! `fp8_tile_blocks(rows) * count` blocks (job-major), dynamic shared `fp8_tile_shared(R, QN*16, D)` bytes.
//! The host launches this body only for `fp8_tile_admits(shape)` with 16-byte aligned weight and input, and
//! `expert_operation` selects it by the block size.
use crate::expert_ops::{Arguments,LaunchArguments};

/// The production geometry.
#[inline(always)]
pub(crate) fn tile(launch:LaunchArguments) {
    project::<{crate::expert_ops::FP8_TILE_ROWS as usize},{(crate::expert_ops::FP8_TILE_COLS/16) as usize},{crate::expert_ops::FP8_TILE_DEPTH as usize},{crate::expert_ops::FP8_TILE_PREFETCH as usize}>(launch)
}

#[inline(always)]
fn project<const R:usize,const QN:usize,const D:usize,const PF:usize>(launch:LaunchArguments) {
    // SAFETY: The host checks every job descriptor (shape, weight, scale and input owners and extents),
    // keeps all owners alive through stream completion, and fills the job table (count entries) for a
    // batch. Weights, scales and inputs are immutable; each (job, row) output word has exactly one writing
    // thread (chain lane l of the block that owns the row) and the output is private and initialized
    // earlier on the stream. The body traps unless the block has 32 + R*QN threads, the dynamic shared
    // region holds the layout below, the descriptor is an admitted Fp8Serial shape with the launch's row
    // count, and the weight and input owners are 16-byte aligned. Global reads stay below rows*cols weight
    // bytes, ceil(rows/block_rows)*ceil(cols/block_cols) scale words and cols input words, because every
    // copy is guarded by row < rows and column < cols (cols is a multiple of 16, a copied item is 16
    // columns, an input pair starts at an even column). Shared addresses stay inside the layout: raw weight
    // stages (R rows of CK bytes), scale stages (R*QN words), raw input stages (CK words), three rounded
    // input buffers (CK words), two product buffers (R rows of CK+4 words). The dynamic region is the only
    // shared allocation in the kernels that reach this body, so it starts at 0. Every shared read follows a write by
    // the thread that reads it or by a thread before a barrier: a producer item (row, q) is processed only when its row
    // is below rows and its 16 columns are below cols, and then every raw, scale and rounded-input word it reads was
    // copied or stored for exactly that item and chunk; the chain lanes read product words only of rows below rows and of
    // whole 16-column groups below cols. Lanes and rows past those ranges read no shared word. The job count is at most
    // FP8_TILE_MAX_JOBS (the sweep covers 1 to 8 jobs). The L2 prefetch reads nothing and only
    // names an address inside the row (guarded by col0+PF*ck<cols).
    // Producer-side shared data written by asynchronous copies is read only by the thread that copied it
    // after its own wait; data shared between threads (rounded input, products) crosses a bar.sync that
    // every thread of the block executes the same number of times (the loop bound depends only on cols).
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
        macro_rules! rounded2 {($first:expr,$second:expr)=>{{
            let (first,second)=($first,$second);let packed:u32;
            asm!("cvt.rn.bf16x2.f32 {packed}, {second}, {first};",packed=out(reg32)packed,first=in(reg32)f32::from_bits(first),second=in(reg32)f32::from_bits(second),options(nomem,nostack));
            [if first&0x7f80_0000==0x7f80_0000 {crate::expert_ops::round(f32::from_bits(first))} else {f32::from_bits(packed<<16)},
                if second&0x7f80_0000==0x7f80_0000 {crate::expert_ops::round(f32::from_bits(second))} else {f32::from_bits(packed&0xffff_0000)}]
        }};}
        macro_rules! store4 {($address:expr,$a:expr,$b:expr,$c:expr,$d:expr)=>{asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",p=in(reg32)$address,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,d=in(reg32)$d)};}
        macro_rules! store2 {($address:expr,$a:expr,$b:expr)=>{asm!("st.shared.v2.b32 [{p}], {{{a}, {b}}};",p=in(reg32)$address,a=in(reg32)$a,b=in(reg32)$b)};}
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        let ck=QN*16;let items=R*QN;let pw=ck+4;
        // Shared layout in bytes.
        let raw_at=0usize;let scale_at=raw_at+D*R*ck;let xraw_at=scale_at+D*items*4;let xr_at=xraw_at+D*ck*4;let prod_at=xr_at+3*ck*4;
        let total=prod_at+2*R*pw*4;
        let tid=_thread_idx_x() as usize;
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        let rows_launch=launch.single.shape.rows as usize;
        if _block_dim_x() as usize!=32+items || D<3 || R>32 || items%4!=0 || (dynamic as usize)<total || launch.count==0 || launch.count>crate::expert_ops::FP8_TILE_MAX_JOBS || rows_launch==0 {asm!("trap;");}
        let per_job=rows_launch.div_ceil(R);let block=_block_idx_x() as usize;
        let (job,row0)=(block/per_job,block%per_job*R);
        if job>=launch.count as usize {return;}
        let p:Arguments=if launch.batch==0 {launch.single} else {(launch.batch as *const Arguments).add(job).read()};
        let shape=p.shape;let rows=shape.rows as usize;let cols=shape.cols as usize;
        let block_cols=shape.block_cols as usize;let block_rows=shape.block_rows as usize;
        if !crate::expert_ops::fp8_tile_admits(&shape) || rows!=rows_launch || (p.weight|p.input[0])&15!=0 {asm!("trap;");}
        let shift=block_cols.trailing_zeros();let blocks=cols.div_ceil(block_cols);
        let chunks=cols.div_ceil(ck);
        let input=p.input[0] as *const f32;let weights=p.weight as *const u8;let scales=p.scales as *const f32;
        let producer=tid>=32;let pt=tid.wrapping_sub(32);
        let (item_row,item_q)=(pt/QN,pt%QN);
        let row=row0+item_row;
        // Producer side: issue the copies of chunk c (always commits one group so the counts stay uniform).
        let issue=|c:usize| {
            if c<chunks {
                let col0=c*ck;let stage=c%D;
                if row<rows && col0+item_q*16<cols {
                    crate::ptx::copy16((raw_at+stage*R*ck+item_row*ck+item_q*16) as u32,weights.add(row*cols+col0+item_q*16).cast());
                    // Pull this row's line PF chunks ahead into L2: one thread per row, so the row's next lines are already
                    // there when their own copies run (DRAM sees longer runs of the row; no change to any value or order).
                    if PF!=0 && item_q==0 && col0+PF*ck<cols {asm!("prefetch.global.L2 [{p}];",p=in(reg64)weights.add(row*cols+col0+PF*ck),options(nostack,readonly));}
                    crate::ptx::copy4((scale_at+stage*items*4+pt*4) as u32,scales.add(row/block_rows*blocks+((col0+item_q*16)>>shift)).cast());
                }
                if pt<ck/2 && col0+pt*2<cols {crate::ptx::copy8((xraw_at+stage*ck*4+pt*8) as u32,input.add(col0+pt*2).cast());}
            }
            crate::ptx::commit();
        };
        // Producer side: round the input pair this thread copied and place it in the physical layout
        // word = slot*4*QN + 4*q + e of column 16*q + 4*slot + e.
        let round_input=|c:usize| {
            if c<chunks && pt<ck/2 && c*ck+pt*2<cols {
                let pair=crate::ptx::load2((xraw_at+(c%D)*ck*4+pt*8) as u32);
                let rounded=rounded2!(pair[0],pair[1]);
                let k=pt*2;let word=((k>>2)&3)*4*QN+(k>>4)*4+(k&3);
                store2!((xr_at+(c%3)*ck*4+word*4) as u32,rounded[0].to_bits(),rounded[1].to_bits());
            }
        };
        // Producer side: the products of chunk c, stored as product[row][slot*4*QN + 4*q + e].
        let produce=|c:usize,buffer:usize| {
            if c<chunks && row<rows && c*ck+item_q*16<cols {
                let stage=c%D;
                let raw=crate::ptx::load4((raw_at+stage*R*ck+item_row*ck+item_q*16) as u32);
                let scale=crate::ptx::load((scale_at+stage*items*4+pt*4) as u32);
                let mut slot=0;
                while slot<4 {
                    let x=crate::ptx::load4((xr_at+(c%3)*ck*4+(slot*4*QN+4*item_q)*4) as u32);
                    let bytes=raw[slot];
                    let (first,second)=(pair!(bytes),pair!(bytes>>16));
                    let p0=mul!(mul!(first[0],scale),f32::from_bits(x[0]));
                    let p1=mul!(mul!(first[1],scale),f32::from_bits(x[1]));
                    let p2=mul!(mul!(second[0],scale),f32::from_bits(x[2]));
                    let p3=mul!(mul!(second[1],scale),f32::from_bits(x[3]));
                    store4!((prod_at+buffer*R*pw*4+item_row*pw*4+(slot*4*QN+4*item_q)*4) as u32,p0.to_bits(),p1.to_bits(),p2.to_bits(),p3.to_bits());
                    slot+=1;
                }
            }
        };
        let mut sum=0.0f32;
        let lane=tid;let own=!producer && lane<R && row0+lane<rows;
        // Prologue: chunks 0..D-2 in flight, inputs 0 and 1 rounded, products of chunk 0 made.
        if producer {
            let mut c=0;while c<D-1 {issue(c);c+=1;}
            pending!(D-3);
            round_input(0);round_input(1);
        }
        crate::ptx::sync();
        if producer {produce(0,0);}
        crate::ptx::sync();
        let mut c=0;
        while c<chunks {
            if producer {
                issue(c+D-1);
                pending!(D-3);
                round_input(c+2);
                produce(c+1,(c+1)&1);
            } else if own {
                let kc=if (c+1)*ck<=cols {ck} else {cols-c*ck};
                let base=(prod_at+(c&1)*R*pw*4+lane*pw*4) as u32;
                let groups=kc/16;
                let mut q=0;
                while q<groups {
                    let a=crate::ptx::load4(base+(q*16) as u32);
                    let b=crate::ptx::load4(base+(q*16+QN*16) as u32);
                    let c2=crate::ptx::load4(base+(q*16+2*QN*16) as u32);
                    let d=crate::ptx::load4(base+(q*16+3*QN*16) as u32);
                    let mut j=0;
                    while j<4 {add!(sum,f32::from_bits(a[j]));j+=1;}
                    j=0;while j<4 {add!(sum,f32::from_bits(b[j]));j+=1;}
                    j=0;while j<4 {add!(sum,f32::from_bits(c2[j]));j+=1;}
                    j=0;while j<4 {add!(sum,f32::from_bits(d[j]));j+=1;}
                    q+=1;
                }
            }
            crate::ptx::sync();
            c+=1;
        }
        if own {*(p.output as *mut f32).add(row0+lane)=crate::expert_ops::round(sum);}
    }
}

