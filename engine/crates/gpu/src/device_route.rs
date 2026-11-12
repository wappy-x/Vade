use crate::expert_ops::{Arguments,Op,Shape};
pub const MAX_EXPERTS:u32=1024;
pub const MAX_PICKS:u32=8;
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Build {pub table:u64,pub ids:u64,pub input:u64,pub quantized:u64,pub quantizers:u64,pub projects:u64,pub output:u64,
    pub experts:u32,pub picks:u32,pub phases:u32,pub down:u32,pub shared:u32,
    /// The pair split. 0: the projector writes one finished word a row (today). 4, 2 or 1: it writes that many RAW
    /// quarter accumulators a row (`nv_cols_q4` / `q2` / `q1`), so a job's output is rows * raw words. Down routes only.
    pub raw:u32}
/// True for the `raw` values of a route: none, or the quarters an owner holds, on a down route.
pub fn raw_admitted(raw:u32,down:u32)->bool {raw==0 || down==1 && (raw==1 || raw==2 || raw==4)}
pub fn selected(value:f32,experts:u32)->Option<u32> {
    if value>=0.0 && value<(experts as f32) && value==(value as u32) as f32 {Some(value as u32)} else {None}
}
pub fn extents(experts:u32,picks:u32,phases:u32)->Option<(usize,usize)> {
    if experts==0 || experts>MAX_EXPERTS || picks==0 || picks>MAX_PICKS || picks>experts || !(1..=2).contains(&phases) {return None;}
    Some(((experts*phases) as usize,(picks*phases) as usize))
}
#[cfg(not(target_arch="nvptx64"))]
pub fn admitted(shapes:&[Shape],experts:u32,picks:u32,phases:u32,owners:&[(u64,u64)])->bool {
    let Some((entries,jobs))=extents(experts,picks,phases) else {return false;};
    let Some(first)=shapes.first() else {return false;};
    shapes.len()==entries && owners.len()==entries && picks==8 && first.cols>=512 && first.cols%512==0
        && crate::quantize_sweep::batch_admitted(first.cols,picks as usize)
        && if crate::expert_ops::nv_tiled(first) {crate::expert_ops::nv_tile_layout_admits(first.rows as usize,first.cols as usize) && jobs as u32<=crate::expert_ops::NV_COLS_MAX_JOBS}
            else {false} // TENSOR-CORE REBASE: the row-major tile kernel keeps the rc.8 chain; such a table takes the host route.
        && shapes.iter().zip(owners).all(|(shape,&(weight,scales))|shape.op==Op::NvProject && shape.tokens==1 && shape.block_rows==first.block_rows
            && shape.rows==first.rows && shape.cols==first.cols && shape.role==first.role && shape.lengths().is_some() && (weight|scales)&15==0)
}
#[cfg(not(target_arch="nvptx64"))]
pub fn sweep()->Vec<(u32,u32,u32)> {
    [1,2,31,32,33,63,64,65,127,128,129,191,255,1024,2048].into_iter().flat_map(|rows|
        [512,1024,2048,4096].into_iter().flat_map(move |cols|[1,2].into_iter().filter_map(move |phases|
            crate::expert_ops::nv_tile_shape_proven(rows,cols,8*phases).then_some((rows,cols,phases))))).collect()
}
#[cfg(not(target_arch="nvptx64"))]
static ACCEPTED:[std::sync::atomic::AtomicU64;3]=[const {std::sync::atomic::AtomicU64::new(0)};3];
#[cfg(not(target_arch="nvptx64"))]
pub fn record(down:bool) {ACCEPTED[usize::from(down)].fetch_add(1,std::sync::atomic::Ordering::Relaxed);}
#[cfg(not(target_arch="nvptx64"))]
pub fn counts()->[u64;3] {ACCEPTED.each_ref().map(|count|count.load(std::sync::atomic::Ordering::Relaxed))}
#[cfg(not(target_arch="nvptx64"))]
pub fn fallback() {ACCEPTED[2].fetch_add(1,std::sync::atomic::Ordering::Relaxed);}
#[inline(always)]
pub fn descriptors(mut projection:Arguments,build:Build,job:u32)->(Arguments,Arguments) {
    let slot=job%build.picks;let quant=if build.shared!=0 {0} else {job};let shape=projection.shape;
    let source=build.input+if build.down!=0 {u64::from(slot)*u64::from(shape.cols)*4} else {0};
    let quantized=build.quantized+u64::from(quant)*u64::from(shape.cols/16*17)*4;
    let words=if build.raw==0 {1} else {u64::from(build.raw)};
    projection.input=[quantized,4,4,4];projection.output=build.output+u64::from(job)*u64::from(shape.rows)*words*4;
    let quantizer=Arguments {input:[source,4,4,4],weight:4,scales:4,output:quantized,shape:Shape {op:Op::Quantize,..shape}};
    (quantizer,projection)
}
const _:()={assert!(core::mem::size_of::<Build>()==80);assert!(core::mem::offset_of!(Build,experts)==56);assert!(core::mem::offset_of!(Build,raw)==76);};
#[cfg(test)]
mod tests {
    use super::*;
    fn shape()->Shape {Shape {op:Op::NvProject,rows:1024,cols:4096,block_rows:crate::expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:0.5,activation:0.25,bound:0.0,role:0}}
    #[test] fn shape_envelope_keeps_unproved_or_unequal_tables_on_the_host() {
        assert!(admitted(&[shape();32],16,8,2,&[(16,32);32]));
        // TENSOR-CORE REBASE: a row-major table takes the host route (its tile kernel keeps the rc.8 chain).
        assert!(!admitted(&[Shape {block_rows:0,..shape()};32],16,8,2,&[(16,32);32]));
        for (experts,picks,phases) in [(0,8,2),(1025,8,2),(4,8,2),(16,9,2),(16,8,3)] {assert!(!admitted(&[shape();32],experts,picks,phases,&[(16,32);32]));}
        let mut shapes=[shape();32];shapes[31].cols=2048;assert!(!admitted(&shapes,16,8,2,&[(16,32);32]));
        assert!(!admitted(&[Shape {rows:258,..shape()};32],16,8,2,&[(16,32);32]));
        assert!(!admitted(&[shape();32],16,8,2,&[(20,32);32]));
        // Review finding U2: one table has one word function. A table of second matrices is a route; a mixed table is not.
        let second=Shape {role:crate::expert_ops::NV_SECOND,..shape()};assert!(admitted(&[second;32],16,8,2,&[(16,32);32]));
        let mut shapes=[shape();32];shapes[31]=second;assert!(!admitted(&shapes,16,8,2,&[(16,32);32]));
        let mut shapes=[second;32];shapes[0]=shape();assert!(!admitted(&shapes,16,8,2,&[(16,32);32]));
    }
    #[test] fn every_admitted_shape_is_in_the_device_sweep() {
        let swept=sweep();
        for rows in 0..=2049 {for cols in [0,128,256,512,1024,1536,2048,2560,3072,3584,4096,4608,8192] {for phases in 1..=2 {
            // TENSOR-CORE REBASE: no row-major table is admitted any more (so none can be outside the sweep);
            // a column-tile table is admitted by its storage rule alone.
            let shape=Shape {rows,cols,block_rows:0,..shape()};let shapes=vec![shape;16*phases as usize];
            if admitted(&shapes,16,8,phases,&vec![(16,32);shapes.len()]) {assert!(swept.contains(&(rows,cols,phases)),"{rows}/{cols}/{phases}");}
            assert!(!admitted(&shapes,16,8,phases,&vec![(16,32);shapes.len()]),"row-major {rows}/{cols}/{phases}");
        }}}
    }
    #[test] fn ids_refuse_nonfinite_fractional_negative_and_outside_before_indexing() {
        for value in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,-1.0,0.5,1024.0] {assert_eq!(selected(value,1024),None);}
        assert_eq!(selected(-0.0,1024),Some(0));assert_eq!(selected(0.0,1024),Some(0));assert_eq!(selected(1023.0,1024),Some(1023));
    }
    #[test] fn role_word_23_stays_inside_every_descriptor_array() {
        use core::mem::{size_of,align_of,offset_of};
        assert_eq!((size_of::<Arguments>(),align_of::<Arguments>(),offset_of!(Arguments,shape)),(96,8,56));
        assert_eq!((size_of::<Shape>(),align_of::<Shape>(),offset_of!(Shape,bound),offset_of!(Shape,role)),(40,4,32,36));
        assert_eq!(offset_of!(Arguments,shape)+size_of::<Shape>(),24*size_of::<u32>());
        let build=Build {table:16,ids:32,input:48,quantized:64,quantizers:80,projects:96,output:112,experts:8,picks:8,phases:1,down:0,shared:0,raw:0};
        let entry=Arguments {input:[4;4],weight:128,scales:144,output:4,shape:shape()};
        let (quantizer,projection)=descriptors(entry,build,7);assert_eq!((quantizer.descriptor_words()[23],projection.descriptor_words()[23]),(0,0));
        // Review finding NEW-1: a down job is a SECOND matrix, and word 23 of both of its descriptors is that role. The
        // device build must store it (it once cleared this word as padding).
        let down=Build {down:1,..build};let second=Arguments {shape:Shape {role:crate::expert_ops::NV_SECOND,..shape()},..entry};
        let (quantizer,projection)=descriptors(second,down,7);
        assert_eq!((quantizer.descriptor_words()[23],projection.descriptor_words()[23]),(crate::expert_ops::NV_SECOND,crate::expert_ops::NV_SECOND));
        assert_eq!((crate::expert_ops::nv_chains(&projection.shape),projection.shape.lengths().is_some(),quantizer.shape.lengths().is_some()),(4,true,true));
        for experts in [8,9,16,255,256,257,1024] {for phases in 1..=2 {for shared in [false,true] {
            let (entries,jobs)=extents(experts,8,phases).unwrap();assert_eq!(entries,experts as usize*phases as usize);
            let quantizers=if shared {1} else {jobs};
            for job in 0..jobs {assert!(job*24+23<jobs*24);if !shared || job==0 {let quant=if shared {0} else {job};assert!(quant*24+23<quantizers*24);}}
        }}}
    }
    #[test] fn original_slots_and_per_expert_calibration_survive_descriptor_selection() {
        let build=Build {table:16,ids:32,input:4096,quantized:32768,quantizers:48,projects:64,output:131072,experts:16,picks:8,phases:1,down:1,shared:0,raw:0};
        let original=Arguments {input:[4;4],weight:80,scales:96,output:4,shape:shape()};
        let (quant,project)=descriptors(original,build,7);
        assert_eq!(quant.input[0],4096+7*4096*4);assert_eq!(project.output,131072+7*1024*4);
        assert_eq!(project.input[0],32768+7*(4096/16*17)*4);assert_eq!(quant.output,project.input[0]);
        assert_eq!(quant.shape.activation.to_bits(),original.shape.activation.to_bits());assert_eq!((project.weight,project.scales),(80,96));
        let (quant,project)=descriptors(original,Build {down:0,shared:1,..build},7);
        assert_eq!(quant.input[0],4096);assert_eq!(quant.output,32768);assert_eq!(project.input[0],32768);
        // The pair split: a raw route steps a job's output by rows * quarters words; its inputs do not move.
        for (raw,words) in [(1u32,1u64),(2,2),(4,4)] {
            let (raw_quant,raw_project)=descriptors(original,Build {raw,..build},7);
            assert_eq!(raw_project.output,131072+7*1024*words*4);assert_eq!((raw_quant.input[0],raw_project.input[0]),(4096+7*4096*4,32768+7*(4096/16*17)*4));
        }
        assert!(raw_admitted(0,0) && raw_admitted(0,1) && raw_admitted(1,1) && raw_admitted(2,1) && raw_admitted(4,1));
        for (raw,down) in [(1,0),(2,0),(4,0),(3,1),(8,1),(4,2)] {assert!(!raw_admitted(raw,down),"{raw}/{down}");}
    }
}
