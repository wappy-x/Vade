//! One FP8 weight row for one token with word loads and hardware E4M3 decode. Each
//! output keeps its ascending fold, so its bits equal `fp8_row_table`.
//!
//! Contract: the only caller is `expert_group`, after the host has checked the input, weight
//! and scale spans of the entry, and each macro runs only behind that kernel's guard.
//! `row` needs cols a multiple of 16, block_cols a power of two of at least 16, weight and
//! input aligned to 16 bytes and scales to 4. Each `ld.global.v4.u32` then reads 16 aligned
//! bytes below cols. `nv_row` needs weight aligned to 4 bytes and the decode table to 8.
//! Each `ld.global.v2.u32` then reads an aligned pair at a byte code below 256, inside the table.
macro_rules! row {($entry:expr,$input:expr,$weight:expr,$scales:expr,$unit:expr,$output:expr)=>{{
    use core::arch::asm;
    macro_rules! load16 {($pointer:expr)=>{{
        let (first,second,third,fourth):(u32,u32,u32,u32);
        asm!("ld.global.v4.u32 {{{first},{second},{third},{fourth}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,third=out(reg32)third,fourth=out(reg32)fourth,pointer=in(reg64)$pointer,options(nostack,readonly));
        [first,second,third,fourth]
    }};}
    macro_rules! pair {($bits:expr)=>{{
        let (halves,first,second):(u32,f32,f32);let bits=$bits as u16;
        asm!("{{ .reg .b16 lo,hi; cvt.rn.f16x2.e4m3x2 {halves}, {bits}; mov.b32 {{lo,hi}}, {halves}; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",halves=out(reg32)halves,first=out(reg32)first,second=out(reg32)second,bits=in(reg16)bits,options(nomem,nostack));
        let _=halves;[first,second]
    }};}
    let shape=$entry.shape;let (cols,tokens)=(shape.cols as usize,shape.tokens as usize);
    let (row,token)=($unit/tokens,$unit%tokens);let block_cols=shape.block_cols as usize;let blocks=cols.div_ceil(block_cols);
    let input=($input as *const f32).add(token*cols);let weight=($weight as *const u8).add(row*cols);
    let scales=($scales as *const f32).add((row/shape.block_rows as usize)*blocks);let shift=block_cols.trailing_zeros();
    let mut sum=0.0f32;let mut column=0;
    while column<cols {
        let weights=load16!(weight.add(column));let scale=*scales.add(column>>shift);let mut word=0;
        while word<4 {
            let values=load16!(input.add(column+word*4).cast::<u8>());let bytes=weights[word];
            let (low,high)=(pair!(bytes),pair!(bytes>>16));
            sum+=(low[0]*scale)*f32::from_bits(values[0]);sum+=(low[1]*scale)*f32::from_bits(values[1]);
            sum+=(high[0]*scale)*f32::from_bits(values[2]);sum+=(high[1]*scale)*f32::from_bits(values[3]);
            word+=1;
        }
        column+=16;
    }
    *$output=crate::expert_ops::round(sum);
}};}
pub(crate) use row;
/// One NVFP4 weight row for one token. E2M1 has no hardware decode on this target, so a
/// byte's two values come from the table in one read, and two words carry a block's bytes.
macro_rules! nv_row {($entry:expr,$input:expr,$weight:expr,$scales:expr,$lut:expr,$unit:expr,$output:expr)=>{{
    use core::arch::asm;
    macro_rules! pair {($pointer:expr)=>{{
        let (first,second):(u32,u32);
        asm!("ld.global.v2.u32 {{{first},{second}}}, [{pointer}];",first=out(reg32)first,second=out(reg32)second,pointer=in(reg64)$pointer,options(nostack,readonly));
        [f32::from_bits(first),f32::from_bits(second)]
    }};}
    let shape=$entry.shape;let (cols,tokens)=(shape.cols as usize,shape.tokens as usize);let blocks=cols/16;
    let (row,token)=($unit/tokens,$unit%tokens);let global=shape.global*shape.activation;
    let input=($input as *const f32).add(token*blocks*17);let weight=($weight as *const u32).add(row*blocks*2);
    let scales=($scales as *const u8).add(row*blocks);let lut=$lut as *const f32;
    let mut sum=0.0f32;let mut block=0;
    while block<blocks {
        let a=input.add(block*17);let fold=(*lut.add(512+*scales.add(block) as usize)**a.add(16))*global;
        let (low,high)=(*weight.add(block*2),*weight.add(block*2+1));
        let mut byte=0;
        while byte<8 {
            let word=if byte<4 {low} else {high};let code=((word>>((byte&3)*8))&255) as usize;let codes=pair!(lut.add(code*2));
            sum+=(codes[0]**a.add(2*byte))*fold;sum+=(codes[1]**a.add(2*byte+1))*fold;
            byte+=1;
        }
        block+=1;
    }
    *$output=crate::expert_ops::round(sum);
}};}
pub(crate) use nv_row;
