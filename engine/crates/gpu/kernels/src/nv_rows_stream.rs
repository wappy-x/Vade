//! NVFP4 batch projection with coalesced weight streaming. Same arithmetic and order as
//! `nv_projection::row!`: every output row is ONE ascending FP32 chain over the columns, in 16-column
//! blocks, each product `(value * activation) * fold` rounded in two steps, no fused multiply-add,
//! the same FP4 decode and the same final BF16 round. What changes is memory access: the previous
//! kernel gave each lane its own row, so a warp load touched 16 rows' lines 8 bytes at a time.
//! Here a block of ROWS rows copies CHUNK_BYTES weight bytes (and the matching scale bytes and
//! activation words) of every row per step into shared memory with asynchronous copies (double
//! buffered), and each thread reads its own row from there.
//!
//! Launch: the same `LaunchArguments` and job table as `expert_operation`. Grid =
//! ceil(rows / ROWS) * count blocks of ROWS threads (job-major). Needs op NvProject,
//! cols % (2 * CHUNK_BYTES) == 0 and 16-byte aligned weight and scale owners.
use crate::expert_ops::{Arguments,LaunchArguments,Op};

const APITCH:usize=80;const SHARED:usize=48128;
#[no_mangle]
pub extern "ptx-kernel" fn nv_rows_stream(launch:LaunchArguments) {body::<64,256>(launch)}

#[inline(always)]
pub(crate) fn body<const ROWS:usize,const CB:usize>(launch:LaunchArguments) {
    // SAFETY: The host checks every job descriptor (shape, weight, scale and input owners and extents)
    // and keeps all owners alive through stream completion; the job table has `count` entries. Weights,
    // scales and inputs are immutable. Each (job, row) output word has exactly one writing thread and
    // the output is private and initialized earlier on the stream. The kernel traps unless op is
    // NvProject, cols is a multiple of 2*CB, both raw owners are 16-byte aligned and the block has ROWS
    // threads. Copies read only rows below `rows` and whole CB-byte weight chunks, CB/8-byte scale
    // chunks and CB/8*17-word activation chunks inside those checked extents. Shared reads and writes
    // stay inside the block's two-stage static array (two stages of at most SHARED/2 bytes). Every
    // shared read follows its write: a thread reads the weight and scale slots of its own row only when
    // that row exists (first_row+tid<rows), and those slots were copied by threads that completed their
    // waits before the barrier that precedes the read; lanes past the last row read no shared slot of a
    // row and do no arithmetic. The activation slots are copied by all threads. All threads reach every
    // barrier; loop bounds depend only on the descriptor.
    unsafe {
        use core::arch::asm;
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        macro_rules! decode {($byte:expr)=>{{
            let (first,second):(f32,f32);let byte=$byte as u32;
            let selector=((byte&7)<<4)|((byte&0x70)<<8);
            let signs=((byte&8)<<12)|((byte&0x80)<<24);
            asm!("{{ .reg .b16 lo,hi; .reg .b32 halves; prmt.b32 halves, 0x3e3c3800, 0x46444240, {selector}; or.b32 halves, halves, {signs}; mov.b32 {{lo,hi}}, halves; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",first=out(reg32)first,second=out(reg32)second,selector=in(reg32)selector,signs=in(reg32)signs,options(nomem,nostack));
            [first,second]
        }};}
        let wpitch=CB+16;let sbytes=CB/8;let nb=CB/8;let actw=nb*17;
        let stage_bytes=ROWS*wpitch+ROWS*sbytes+nb*APITCH;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let threads=core::arch::nvptx::_block_dim_x() as usize;
        if threads!=ROWS || launch.batch==0 || launch.count==0 || 2*stage_bytes>SHARED {core::arch::asm!("trap;");}
        let rows=launch.single.shape.rows as usize;let cols=launch.single.shape.cols as usize;
        let per_job=rows.div_ceil(ROWS);let block=core::arch::nvptx::_block_idx_x() as usize;
        let (job,first_row)=(block/per_job,block%per_job*ROWS);
        if job>=launch.count as usize {return;}
        let p:Arguments=(launch.batch as *const Arguments).add(job).read();
        if p.shape.op!=Op::NvProject || cols%(2*CB)!=0 || p.shape.rows as usize!=rows || p.shape.cols as usize!=cols || (p.weight|p.scales)&15!=0 {core::arch::asm!("trap;");}
        let smem=crate::ptx::shared::<SHARED>();
        let input=p.input[0] as *const f32;let weights=p.weight as *const u8;let scales=p.scales as *const u8;
        let chunks=cols/(2*CB);let row_bytes=cols/2;let scale_row=cols/16;
        let issue=|c:usize| {
            let stage=smem+((c&1)*stage_bytes) as u32;
            let mut item=tid;
            while item<ROWS*(CB/16) {let (r,q)=(item/(CB/16),item%(CB/16));
                if first_row+r<rows {
                    crate::ptx::copy16(stage+(r*wpitch+q*16) as u32,weights.add((first_row+r)*row_bytes+c*CB+q*16).cast());}
                item+=ROWS;}
            let mut item=tid;
            while item<ROWS*(sbytes/16) {let (r,q)=(item/(sbytes/16),item%(sbytes/16));
                if first_row+r<rows {crate::ptx::copy16(stage+(ROWS*wpitch+r*sbytes+q*16) as u32,scales.add((first_row+r)*scale_row+c*sbytes+q*16).cast());}
                item+=ROWS;}
            let mut word=tid;
            while word<actw {let (b,j)=(word/17,word%17);crate::ptx::copy4(stage+(ROWS*wpitch+ROWS*sbytes+b*APITCH+j*4) as u32,input.add(c*actw+word).cast());word+=ROWS;}
        };
        let global=mul!(p.shape.global,p.shape.activation);
        let mut sum=0.0f32;
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*stage_bytes) as u32;
            let row_words=stage+(tid*wpitch) as u32;let scale_words=stage+(ROWS*wpitch+tid*sbytes) as u32;
            let acts=stage+(ROWS*wpitch+ROWS*sbytes) as u32;
            // Lanes past the last row have no copied weights or scales: they read nothing and add nothing (the barriers below stay uniform).
            if first_row+tid<rows {
                let mut b=0;
                while b<nb {
                    let scale_byte=crate::ptx::load_u8(scale_words+b as u32);
                    let activation_scale=crate::ptx::load(acts+(b*APITCH+64) as u32);
                    let fold=mul!(mul!(crate::expert_ops::e4m3(scale_byte as u8),activation_scale),global);
                    let words=crate::ptx::load2(row_words+(b*8) as u32);
                    let x0=crate::ptx::load4(acts+(b*APITCH) as u32);let x1=crate::ptx::load4(acts+(b*APITCH+16) as u32);
                    let x2=crate::ptx::load4(acts+(b*APITCH+32) as u32);let x3=crate::ptx::load4(acts+(b*APITCH+48) as u32);
                    let x=[x0,x1,x2,x3];
                    let mut pair=0;
                    while pair<8 {
                        let values=decode!((words[pair/4]>>((pair%4)*8))&255);
                        let a0=f32::from_bits(x[pair/2][(pair%2)*2]);let a1=f32::from_bits(x[pair/2][(pair%2)*2+1]);
                        add!(sum,mul!(mul!(values[0],a0),fold));
                        add!(sum,mul!(mul!(values[1],a1),fold));
                        pair+=1;
                    }
                    b+=1;
                }
            }
            crate::ptx::sync();
            c+=1;
        }
        if first_row+tid<rows {*(p.output as *mut f32).add(first_row+tid)=crate::expert_ops::round(sum);}
    }
}
