//! One FP8 weight row for one token with word loads and hardware E4M3 decode. Each
//! output keeps its ascending fold, so its bits equal `fp8_row_table`.
//!
//! Contract: the only caller is `expert_group`, after the host has checked the input, weight
//! and scale spans of the entry, and each macro runs only behind that kernel's guard.
//! `row` needs cols a multiple of 16, block_cols a power of two of at least 16, weight and
//! input aligned to 16 bytes and scales to 4. Each `ld.global.v4.u32` then reads 16 aligned
//! bytes below cols.
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
