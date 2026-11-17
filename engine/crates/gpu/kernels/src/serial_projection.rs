macro_rules! plain_row {($weights:expr,$input:expr,$cols:expr)=>{{
    let (weights,input,cols)=($weights,$input,$cols);let mut sum=0.0f32;
    macro_rules! accumulate {($weight:expr,$value:expr)=>{{
        let product:f32;core::arch::asm!("mul.rn.f32 {product}, {weight}, {value};",product=out(reg32)product,weight=in(reg32)$weight,value=in(reg32)$value,options(nomem,nostack));
        core::arch::asm!("add.rn.f32 {sum}, {sum}, {product};",sum=inout(reg32)sum,product=in(reg32)product,options(nomem,nostack));
    }};}
    if cols%4==0 && (weights as usize|input as usize)&15==0 {
        let mut column=0;while column<cols {
            let (first,second,third,fourth):(f32,f32,f32,f32);let (input_first,input_second,input_third,input_fourth):(f32,f32,f32,f32);
            core::arch::asm!("ld.global.v4.f32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,third=out(reg32)third,fourth=out(reg32)fourth,pointer=in(reg64)weights.add(column));
            core::arch::asm!("ld.global.v4.f32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)input_first,second=out(reg32)input_second,third=out(reg32)input_third,fourth=out(reg32)input_fourth,pointer=in(reg64)input.add(column));
            accumulate!(first,input_first);accumulate!(second,input_second);accumulate!(third,input_third);accumulate!(fourth,input_fourth);column+=4;
        }
    } else {let mut column=0;while column<cols {accumulate!(*weights.add(column),*input.add(column));column+=1;}}
    sum
}};}
pub(crate) use plain_row;
macro_rules! project {($launch:expr,$thread:expr)=>{{
        use crate::expert_ops::{Arguments,Op};
        let (launch,thread)=($launch,$thread);
        let Some((job,row))=crate::expert_ops::serial_projection_index(launch.single.shape.rows,launch.count,thread) else {return;};
        let parameters=if launch.batch==0 {launch.single} else {(launch.batch as *const Arguments).add(job).read()};
        let shape=parameters.shape;let cols=shape.cols as usize;let input=parameters.input[0] as *const f32;
        if shape.op==Op::PlainSerial {
            let weight=(parameters.weight as *const f32).add(row*cols);
            *(parameters.output as *mut f32).add(row)=crate::serial_projection::plain_row!(weight,input,cols);
        } else {
            // TENSOR-CORE REBASE (block FP8): the one-warp body gives the contract word of `tc_f8` through the shared
            // `expert_ops::unit` (a raw quarter job included). The word-load chain kept the old one-chain word and is gone.
            let Some((lengths,weight_bytes,scale_bytes,_,_))=shape.lengths() else {return;};
            let x=core::slice::from_raw_parts(input,lengths[0]);
            let w=core::slice::from_raw_parts(parameters.weight as *const u8,weight_bytes);
            let scales=core::slice::from_raw_parts(parameters.scales as *const u8,scale_bytes);
            let mut out=[0.0f32];crate::expert_ops::unit(shape,row,[x,&[],&[],&[]],w,scales,&mut out);
            *(parameters.output as *mut f32).add(row)=out[0];
        }
}};}
pub(crate) use project;
