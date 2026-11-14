//! Register expansion of one packed NV row. The caller owns its output slice.
macro_rules! row {($shape:expr,$index:expr,$input:expr,$weights:expr,$scales:expr)=>{{
    use core::arch::asm;
    let (shape,index,input,weights,scales)=($shape,$index,$input,$weights,$scales);
    macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
    macro_rules! add {($sum:ident,$value:expr)=>{{let value=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
    macro_rules! decode {($byte:expr)=>{{
        let (first,second):(f32,f32);let byte=$byte as u32;
        // The high bytes of the eight positive half values are
        // 00,38,3c,3e,40,42,44,46. Both low bytes are zero. PRMT places
        // these bytes in two half words; OR inserts each independent sign.
        let selector=((byte&7)<<4)|((byte&0x70)<<8);
        let signs=((byte&8)<<12)|((byte&0x80)<<24);
        asm!("{{ .reg .b16 lo,hi; .reg .b32 halves; prmt.b32 halves, 0x3e3c3800, 0x46444240, {selector}; or.b32 halves, halves, {signs}; mov.b32 {{lo,hi}}, halves; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",first=out(reg32)first,second=out(reg32)second,selector=in(reg32)selector,signs=in(reg32)signs,options(nomem,nostack));
        [first,second]
    }};}
    let cols=shape.cols as usize;let global=mul!(shape.global,shape.activation);let mut sum=0.0f32;
    let weight=weights.as_ptr().add(index*(cols/2));let mut block=0;
    while block<cols/16 {
        let activation=&input[block*17..block*17+17];
        let fold=mul!(mul!(crate::expert_ops::e4m3(scales[index*(cols/16)+block]),activation[16]),global);
        let pointer=weight.add(block*8);
        if pointer as usize&7==0 {
            let (low,high):(u32,u32);
            asm!("ld.global.v2.u32 {{{low},{high}}}, [{pointer}];",low=out(reg32)low,high=out(reg32)high,pointer=in(reg64)pointer);
            let words=[low,high];let mut pair=0;
            while pair<8 {
                let values=decode!((words[pair/4]>>((pair%4)*8))&255);
                add!(sum,mul!(mul!(values[0],activation[2*pair]),fold));
                add!(sum,mul!(mul!(values[1],activation[2*pair+1]),fold));pair+=1;
            }
        } else {let mut pair=0;while pair<8 {
            let values=decode!(*pointer.add(pair));
            add!(sum,mul!(mul!(values[0],activation[2*pair]),fold));
            add!(sum,mul!(mul!(values[1],activation[2*pair+1]),fold));pair+=1;
        }}block+=1;
    }crate::expert_ops::round(sum)
}};}
pub(crate) use row;
