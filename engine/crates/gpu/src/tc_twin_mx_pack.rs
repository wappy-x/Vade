//! CPU twin of `mx_tc_quantize_pack` (kernels `mx_tc.rs`): the packed token rows of an MXFP8 prompt group made from the
//! RAW rows in one launch. The reference is the two launches it stands for: the quantizer's decoded lanes
//! (`mxfp8_quantize::lane`, plane 0), then the twin's packed row (`tc_twin_mx::pack_row`). `quantize_pack_group` is the
//! kernel's own way, step for step; the test shows it gives the reference's bytes.
use crate::tc_twin_mx::{activation_scale_code,e4m3_code,encode_group,operand_group,pack_row};
/// The packed token row of a prompt group, from its RAW rows: `pack_row` of the quantizer's decoded lanes.
pub fn quantize_pack_row(input:&[f32])->Vec<u8> {
    let decoded:Vec<f32>=(0..input.len()).map(|index|crate::mxfp8_quantize::lane(input,index,0)).collect();
    pack_row(&decoded)
}
/// The E4M3 code of 224 and the last scale code of a finite group (kernels `mx_tc::CODE_224`, `LAST_SCALE`).
pub const CODE_224:u8=118;
pub const LAST_SCALE:u8=247;
/// One group of `mx_tc_quantize_pack`, step for step as the kernel takes it: (scale code, the 32 codes). The short way
/// (every source finite, the scale code below `LAST_SCALE`): the codes of the BF16-rounded sources at their own scale
/// code; when the largest lands on 224 and the scale code is not 0, one scale code lower with every code doubled.
/// Every other group: the quantizer's lanes, then the pack's operand (the codes of a group that is not finite are not
/// part of the twin's contract).
pub fn quantize_pack_group(input:&[f32])->(u8,[u8;32]) {
    assert_eq!(input.len(),32);
    let sources:Vec<f32>=input.iter().map(|&x|crate::bf16::rounded(x)).collect();
    let maximum=sources.iter().map(|x|x.to_bits()&0x7fff_ffff).max().unwrap();
    let mut code=activation_scale_code(maximum);
    if maximum>=0x7f80_0000 || code>=LAST_SCALE {
        let decoded:Vec<f32>=(0..32).map(|index|crate::mxfp8_quantize::lane(input,index,0)).collect();
        return operand_group(&decoded);
    }
    let (_,_,mut codes)=encode_group(&sources);
    let top=e4m3_code(f32::from_bits(maximum)*f32::from_bits((254-u32::from(code))<<23))&0x7f;
    if code!=0 && top==CODE_224 {
        code-=1;
        for slot in &mut codes {let magnitude=*slot&0x7f;*slot=(*slot&0x80)|if magnitude<8 {2*magnitude} else {magnitude+8};}
    }
    (code,codes)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tc_twin_mx::{NOT_FINITE,Rng,activation_group};
    /// Groups around the quantizer's edges (the generator of kernel-bench --mx-quant pack, kinds 5 to 7): the largest value
    /// of a group has the exponent field `low + below(span)`; a quarter of the groups put it in (224, 232] at its scale.
    fn edge_group(r:&mut Rng,low:u32,span:u64)->[f32;32] {
        let s=r.next_u64();let field=low+(s%span) as u32;
        let top=match (s>>8)%4 {0=>97+((s>>16)%8) as u32,1=>96,2=>105+((s>>16)%23) as u32,_=>((s>>16)%128) as u32};
        let place=(s>>32) as usize%32;let mut group=[0.0f32;32];
        for (lane,slot) in group.iter_mut().enumerate() {
            let t=r.next_u64();
            let bits=if lane==place {field<<23|top<<16} else if (t>>20).is_multiple_of(16) {0} else {
                let drop=(t%16) as u32;let fraction=((t>>8)%128) as u32;
                field.saturating_sub(drop)<<23|(if drop==0 {fraction.min(top)} else {fraction})<<16};
            let low=if (t>>44).is_multiple_of(8) {(t>>48) as u32&0xffff} else {0};
            *slot=f32::from_bits(bits|low|if (t>>40)&1==1 {0x8000_0000} else {0});
        }
        group
    }
    #[test]
    fn quantize_pack_takes_the_short_way_to_the_same_bytes() {
        let mut r=Rng(0x51ed_270b);
        // [groups, not finite, of those with finite sources, the scale code one below the quantizer's, quantizer scale code 0]
        let mut edges=[0usize;5];
        for round in 0..60_000usize {
            let group=match round%6 {0=>activation_group(&mut r,0,round/6),1|2=>edge_group(&mut r,9,242),3=>edge_group(&mut r,250,5),4=>edge_group(&mut r,0,13),
                _=>{let mut g=edge_group(&mut r,100,60);if round%600==5 {g[round%32]=if round%1200==5 {f32::NAN} else {f32::NEG_INFINITY};}g}};
            let row=quantize_pack_row(&group);let (code,codes)=quantize_pack_group(&group);
            assert_eq!((code,&codes[..]),(row[32],&row[..32]),"round {round} group {:08x?}",group.map(f32::to_bits));
            let own=crate::mxfp8_quantize::lane(&group,0,2);
            edges[0]+=1;
            if code==NOT_FINITE {edges[1]+=1;if own.is_finite() {edges[2]+=1;}} else if f32::from(code)+1.0==own {edges[3]+=1;}
            if own==0.0 {edges[4]+=1;}
        }
        // Every edge is reached many times, the short way's own (the scale code one below) most of all.
        assert!(edges[1]>100 && edges[2]>100 && edges[3]>5_000 && edges[4]>100,"{edges:?}");
        // By hand: the largest value 1.8125 is 232 at scale code 120, a tie that goes to the even code, 224. The pack then
        // takes scale code 119 and writes 448 (0x7e). -0.4375 is -56 (0x66 with the sign), doubled -112 (0xee). 2^-17 is
        // half the smallest step: a tie to code 0, which stays 0 (quantized straight at scale code 119 it would be code 1:
        // the two rounds are kept). 3 x 2^-16 and 5 x 2^-16 are the subnormal codes 3 and 5, doubled 6 and 10.
        let mut group=[0.0f32;32];group[..6].copy_from_slice(&[1.8125,-0.4375,1.0/131072.0,-0.0,3.0/65536.0,5.0/65536.0]);
        let (code,codes)=quantize_pack_group(&group);
        assert_eq!(code,119);assert_eq!(&codes[..6],&[0x7e,0xee,0x00,0x80,0x06,0x0a]);
        assert_eq!((code,&codes[..]),(quantize_pack_row(&group)[32],&quantize_pack_row(&group)[..32]));
    }
}
