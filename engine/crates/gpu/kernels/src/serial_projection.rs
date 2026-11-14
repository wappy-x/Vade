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
        use core::arch::asm;
        use crate::expert_ops::{Arguments,Op};
        let (launch,thread)=($launch,$thread);
        macro_rules! load16 {($pointer:expr)=>{{
            let (first,second,third,fourth):(u32,u32,u32,u32);
            asm!("ld.global.v4.u32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,third=out(reg32)third,fourth=out(reg32)fourth,pointer=in(reg64)$pointer);
            [first,second,third,fourth]
        }};}
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
        macro_rules! mul {($first:expr,$second:expr)=>{{let product:f32;asm!("mul.rn.f32 {product}, {first}, {second};",product=out(reg32)product,first=in(reg32)$first,second=in(reg32)$second,options(nomem,nostack));product}};}
        macro_rules! add {($sum:ident,$product:expr)=>{{let product=$product;asm!("add.rn.f32 {sum}, {sum}, {product};",sum=inout(reg32)$sum,product=in(reg32)product,options(nomem,nostack));}};}
        let Some((job,row))=crate::expert_ops::serial_projection_index(launch.single.shape.rows,launch.count,thread) else {return;};
        let parameters=if launch.batch==0 {launch.single} else {(launch.batch as *const Arguments).add(job).read()};
        let shape=parameters.shape;let cols=shape.cols as usize;let input=parameters.input[0] as *const f32;let mut sum=0.0f32;
        if shape.op==Op::PlainSerial {
            let weight=(parameters.weight as *const f32).add(row*cols);
            *(parameters.output as *mut f32).add(row)=crate::serial_projection::plain_row!(weight,input,cols);
        } else {
            let block_cols=shape.block_cols as usize;let blocks=cols.div_ceil(block_cols);
            let weight=(parameters.weight as *const u8).add(row*cols);
            let scales=(parameters.scales as *const f32).add(row/shape.block_rows as usize*blocks);
            if cols%16==0 && block_cols>=16 && block_cols.is_power_of_two() && (parameters.weight|parameters.input[0])&15==0 {
                let shift=block_cols.trailing_zeros();let mut column=0;
                while column<cols {
                    let weights=load16!(weight.add(column));let scale=*scales.add(column>>shift);let mut word=0;
                    while word<4 {
                        let values=load16!(input.add(column+word*4).cast::<u8>());let bytes=weights[word];
                        let (first,second)=(pair!(bytes),pair!(bytes>>16));let (low,high)=(rounded2!(values[0],values[1]),rounded2!(values[2],values[3]));
                        add!(sum,mul!(mul!(first[0],scale),low[0]));add!(sum,mul!(mul!(first[1],scale),low[1]));
                        add!(sum,mul!(mul!(second[0],scale),high[0]));add!(sum,mul!(mul!(second[1],scale),high[1]));word+=1;
                    }column+=16;
                }
            } else {let mut column=0;while column<cols {add!(sum,mul!(mul!(crate::expert_ops::e4m3(*weight.add(column)),*scales.add(column/block_cols)),crate::expert_ops::round(*input.add(column))));column+=1;}}
            *(parameters.output as *mut f32).add(row)=crate::expert_ops::round(sum);
        }
}};}
pub(crate) use project;
