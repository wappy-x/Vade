//! Host checks of the MXFP8 tensor-core twin against the engine's own decoder and activation quantizer. They live
//! outside `tc_twin_mx.rs` because that file is also built by path in the bench tool, which has neither module.
use crate::tc_twin_mx::*;
#[test]
fn the_twin_decodes_a_code_as_the_engine_does() {
    for code in 0..=255u8 {if code&0x7f!=0x7f {assert_eq!(e4m3_value(code).to_bits(),crate::expert_ops::e4m3(code).to_bits(),"decoder {code}");}}
}
#[test]
fn a_quantized_group_is_encoded_exactly_and_the_scale_is_the_quantizer_rule() {
    let mut r=Rng(5);
    for round in 0..20_000 {
        let raw:Vec<f32>=(0..32).map(|_|{let h=r.next_u64();f32::from_bits((h as u32&0x807f_ffff)|((90+(h>>40) as u32%70)<<23))}).collect();
        for positive_zero in [false,true] {
            let lane=|index:usize,plane:u32|if positive_zero {crate::mxfp8_quantize::positive_zero_lane(&raw,index,plane)} else {crate::mxfp8_quantize::lane(&raw,index,plane)};
            let quantized:Vec<f32>=(0..32).map(|i|lane(i,0)).collect();
            let (code,factor,codes)=encode_group(&quantized);
            // The same 32 numbers: every value is its code times the scale, sign of zero included.
            for i in 0..32 {assert_eq!((e4m3_value(codes[i])*factor).to_bits(),quantized[i].to_bits(),"round {round} index {i}");}
            // The quantizer's own scale code, or one less when its largest code is 224 (then the doubled codes fit).
            let own=lane(0,2) as u8;
            assert!(code==own || code+1==own,"round {round}: {code} against {own}");
            // Encoding the encoded group again gives the same operand.
            let again:Vec<f32>=codes.iter().map(|&c|e4m3_value(c)*factor).collect();
            assert_eq!(encode_group(&again),(code,factor,codes));
        }
    }
}
