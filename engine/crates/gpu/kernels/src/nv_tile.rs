//! NVFP4 decode batch projection, shared-memory tile body (lane F's block-FP8 tile method, ported to FP4).
//! Same arithmetic and order as `nv_projection::row!` / `nv_rows_stream`: every output row is ONE ascending
//! FP32 chain over its columns from +0.0, in 16-column blocks, each product `(value * activation) * fold` made by
//! two separate `mul.rn.f32` with `fold = (e4m3(scale) * activation_scale) * (global * activation)`, each step one
//! separate `add.rn.f32` (no fused multiply-add), the same FP4 decode and the same final BF16 round.
//! What changes is who does the work. A block owns R rows. R*QN producer threads copy the weights, the block scales
//! and the quantized activations with asynchronous copies (D stages deep), decode the weights and make the products
//! into shared memory; one chain lane per row (warp 0) only adds the products of its row in column order while the
//! producers make the next chunk of QN*16 columns.
//!
//! Launch: the same `LaunchArguments` and job table as `expert_operation`; blocks of 32 + R*QN threads,
//! `ceil(rows/R) * count` blocks (job-major), dynamic shared `shared_bytes(R, QN, D)` bytes.
use crate::expert_ops::{Arguments,LaunchArguments,Op};

const APITCH_WORDS:usize=20;

/// Dynamic shared bytes of one block: D stages of raw weights (R rows of QN*8 bytes), raw scales (R rows of QN bytes
/// padded to 8), raw activations (QN blocks of 20 words), and two product buffers (R rows of QN*16+4 words).
pub const fn shared_bytes(r:usize,qn:usize,d:usize)->usize {
    d*(r*qn*8+r*(qn.div_ceil(8)*8)+qn*APITCH_WORDS*4)+2*r*(qn*16+4)*4
}

#[no_mangle]
pub extern "ptx-kernel" fn nv_tile(launch:LaunchArguments) {project::<16,8,4,2>(launch)}

#[inline(always)]
pub(crate) fn project<const R:usize,const QN:usize,const D:usize,const PF:usize>(launch:LaunchArguments) {
    // SAFETY: The host checks every job descriptor (shape, weight, scale and input owners and extents), keeps all
    // owners alive through stream completion and fills the job table (count entries). Weights, scales and inputs
    // are immutable; each (job, row) output word has exactly one writing thread (chain lane l of the block that owns
    // the row, a raw store before any read of it) and the output is private; the host skips its fill (fill_free). The body traps unless the block has
    // 32 + R*QN threads, the dynamic shared region holds the layout below, the descriptor is an NvProject job with the
    // launch's row count, cols is a multiple of QN*16 and of 128, and the weight and scale owners are 16-byte
    // aligned and the input owner 4-byte aligned. Every copy is guarded by row < rows and reads inside the checked extents: 8 weight bytes inside the
    // row (cols/2 bytes), 8 scale bytes inside the row (cols/16 bytes, a multiple of 8) and 4-byte activation words
    // below cols/16*17. Shared addresses stay inside the layout: D stages of raw weights, scales and activations and
    // two product buffers, all below `total` bytes. The dynamic region is the only shared allocation in this kernel,
    // so it starts at 0. The L2 prefetch reads nothing and names an address inside the row (guarded by
    // col0+PF*ck<cols). Asynchronous-copy data is read by another thread only after the copier's own wait and a
    // bar.sync that every thread of the block executes the same number of times (the loop bound depends only on cols).
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        macro_rules! decode {($byte:expr)=>{{
            let (first,second):(f32,f32);let byte=$byte as u32;
            let selector=((byte&7)<<4)|((byte&0x70)<<8);
            let signs=((byte&8)<<12)|((byte&0x80)<<24);
            asm!("{{ .reg .b16 lo,hi; .reg .b32 halves; prmt.b32 halves, 0x3e3c3800, 0x46444240, {selector}; or.b32 halves, halves, {signs}; mov.b32 {{lo,hi}}, halves; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",first=out(reg32)first,second=out(reg32)second,selector=in(reg32)selector,signs=in(reg32)signs,options(nomem,nostack));
            [first,second]
        }};}
        macro_rules! store4 {($address:expr,$a:expr,$b:expr,$c:expr,$d:expr)=>{asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",p=in(reg32)$address,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,d=in(reg32)$d)};}
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        let ck=QN*16;let items=R*QN;let pw=ck+4;let sc=QN.div_ceil(8)*8;
        // Shared layout in bytes.
        let raw_at=0usize;let scale_at=raw_at+D*R*QN*8;let act_at=scale_at+D*R*sc;let prod_at=act_at+D*QN*APITCH_WORDS*4;
        let total=prod_at+2*R*pw*4;
        let tid=_thread_idx_x() as usize;
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        let rows_launch=launch.single.shape.rows as usize;let cols_launch=launch.single.shape.cols as usize;
        if _block_dim_x() as usize!=32+items || D<3 || R>32 || QN%8!=0 || (dynamic as usize)<total || launch.count==0 || launch.batch==0 || rows_launch==0 || cols_launch%128!=0 || cols_launch%ck!=0 {asm!("trap;");}
        let per_job=rows_launch.div_ceil(R);let block=_block_idx_x() as usize;
        let (job,row0)=(block/per_job,block%per_job*R);
        if job>=launch.count as usize {return;}
        let p:Arguments=(launch.batch as *const Arguments).add(job).read();
        let shape=p.shape;let rows=shape.rows as usize;let cols=shape.cols as usize;
        if shape.op!=Op::NvProject || rows!=rows_launch || cols!=cols_launch || (p.weight|p.scales)&15!=0 || p.input[0]&3!=0 {asm!("trap;");}
        let chunks=cols/ck;let row_bytes=cols/2;let scale_row=cols/16;
        let input=p.input[0] as *const f32;let weights=p.weight as *const u8;let scales=p.scales as *const u8;
        let producer=tid>=32;let pt=tid.wrapping_sub(32);
        let (item_row,item_q)=(pt/QN,pt%QN);
        let row=row0+item_row;
        let global=mul!(shape.global,shape.activation);
        // Producer side: issue the copies of chunk c (always commits one group so the counts stay uniform).
        let issue=|c:usize| {
            if c<chunks {
                let col0=c*ck;let stage=c%D;
                if row<rows {
                    crate::ptx::copy8((raw_at+stage*R*QN*8+item_row*QN*8+item_q*8) as u32,weights.add(row*row_bytes+col0/2+item_q*8).cast());
                    if item_q<QN/8 {crate::ptx::copy8((scale_at+stage*R*sc+item_row*sc+item_q*8) as u32,scales.add(row*scale_row+col0/16+item_q*8).cast());}
                    // Pull this row's line PF chunks ahead into L2: one thread per row, so the row's next lines are already there
                    // when their own copies run (no change to any value or order).
                    if PF!=0 && item_q==0 && col0+PF*ck<cols {asm!("prefetch.global.L2 [{p}];",p=in(reg64)weights.add(row*row_bytes+(col0+PF*ck)/2),options(nostack,readonly));}
                }
                let mut word=pt;
                while word<QN*17 {
                    let (b,j)=(word/17,word%17);
                    crate::ptx::copy4((act_at+stage*QN*APITCH_WORDS*4+b*APITCH_WORDS*4+j*4) as u32,input.add(c*QN*17+word).cast());
                    word+=items;
                }
            }
            crate::ptx::commit();
        };
        // Producer side: the products of chunk c, stored as product[row][slot*4*QN + 4*q + e] for column 16*q + 4*slot + e.
        let produce=|c:usize,buffer:usize| {
            if c<chunks && row<rows {
                let stage=c%D;
                let words=crate::ptx::load2((raw_at+stage*R*QN*8+item_row*QN*8+item_q*8) as u32);
                let scale_byte=crate::ptx::load_u8((scale_at+stage*R*sc+item_row*sc+item_q) as u32);
                let act=(act_at+stage*QN*APITCH_WORDS*4+item_q*APITCH_WORDS*4) as u32;
                let activation_scale=crate::ptx::load(act+64);
                let fold=mul!(mul!(crate::expert_ops::e4m3(scale_byte as u8),activation_scale),global);
                let mut slot=0;
                while slot<4 {
                    let x=crate::ptx::load4(act+(slot*16) as u32);
                    let bytes=words[slot/2]>>((slot%2)*16);
                    let first=decode!(bytes&255);let second=decode!((bytes>>8)&255);
                    let p0=mul!(mul!(first[0],f32::from_bits(x[0])),fold);
                    let p1=mul!(mul!(first[1],f32::from_bits(x[1])),fold);
                    let p2=mul!(mul!(second[0],f32::from_bits(x[2])),fold);
                    let p3=mul!(mul!(second[1],f32::from_bits(x[3])),fold);
                    store4!((prod_at+buffer*R*pw*4+item_row*pw*4+(slot*4*QN+4*item_q)*4) as u32,p0.to_bits(),p1.to_bits(),p2.to_bits(),p3.to_bits());
                    slot+=1;
                }
            }
        };
        let mut sum=0.0f32;
        let lane=tid;let own=!producer && lane<R && row0+lane<rows;
        // Prologue: chunks 0..D-2 in flight, chunks 0 and 1 complete, products of chunk 0 made.
        if producer {
            let mut c=0;while c<D-1 {issue(c);c+=1;}
            pending!(D-3);
        }
        crate::ptx::sync();
        if producer {produce(0,0);}
        crate::ptx::sync();
        let mut c=0;
        while c<chunks {
            if producer {
                issue(c+D-1);
                pending!(D-3);
                produce(c+1,(c+1)&1);
            } else if own {
                let base=(prod_at+(c&1)*R*pw*4+lane*pw*4) as u32;
                let mut q=0;
                while q<QN {
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
