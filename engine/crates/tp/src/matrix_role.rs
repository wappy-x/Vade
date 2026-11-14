//! The role of a matrix in its pair, and the NVFP4 product of the host fallback.
//!
//! A pair of matrices maps the model width to an inner width and back. The FIRST matrix reads the model width
//! (expert gate and up, q, k, v, routers). The SECOND matrix writes it back (expert down, shared down, dense down,
//! the attention output projections). The role fixes the word function of an NVFP4 product (`tc_contract`):
//! one chain a word for a first matrix, quarter chains for a second matrix. Every path that builds an NVFP4
//! shape asks here, so a matrix has one role on every path.
use crate::Result;
use engine_model::flash::{activation_nvfp4::Activation,nvfp4};
use spark_gpu_probe::expert_ops::{self,Op,Shape,NV_FIRST,NV_SECOND};
/// True for a second matrix, by the last name part of the weight: `down_proj`, `o_proj` or `out_proj`.
pub fn second_matrix(name:&str)->bool {
    matches!(name.strip_suffix(".weight").unwrap_or(name).rsplit('.').next(),Some("down_proj"|"o_proj"|"out_proj"))
}
/// The role word of an NVFP4 shape (`expert_ops::Shape::role`) for this weight.
pub fn nv_role(name:&str)->u32 {if second_matrix(name) {NV_SECOND} else {NV_FIRST}}
/// One NVFP4 matrix times one activation row on the host, from the checkpoint's packed bytes: the contract word of
/// the matrix role (`expert_ops::nv_word`), the word the resident kernels give. `weight` holds two codes a byte,
/// row-major; `scales` one block-scale code for 16 columns. The activation is quantized with the calibrated
/// `input_scale` exactly as before (`Activation::from_bf16_source`).
pub fn nv_host_product(rows:usize,cols:usize,weight:&[u8],scales:&[u8],global_scale:f32,input_scale:f32,role:u32,input:&[f32])->Result<Vec<f32>> {
    if input.len()!=cols {return Err("NVFP4 host product input width differs".into());}
    let activation=Activation::from_bf16_source(input,input_scale).map_err(|e|e.to_string())?;
    let mut quantized=Vec::with_capacity(cols/16*17);
    for block in 0..cols/16 {
        for column in block*16..block*16+16 {quantized.push(nvfp4::e2m1(activation.codes()[column/2]>>((column%2)*4)));}
        quantized.push(nvfp4::e4m3(activation.scales()[block]));
    }
    let shape=Shape {op:Op::NvProject,rows:rows.try_into().map_err(|_|"NVFP4 host product rows exceed ABI")?,cols:cols.try_into().map_err(|_|"NVFP4 host product columns exceed ABI")?,
        block_rows:1,block_cols:16,tokens:1,global:global_scale,activation:activation.global(),bound:0.0,role};
    expert_ops::reference(shape,[&quantized,&[],&[],&[]],weight,scales)
}
#[cfg(test)]
mod tests {
    use super::*;
    use spark_gpu_probe::tc_twin;
    #[test]
    fn the_role_follows_the_last_name_part_of_the_weight() {
        for name in ["model.layers.3.mlp.experts.7.down_proj.weight","model.layers.3.mlp.shared_experts.down_proj.weight","model.layers.0.mlp.down_proj.weight",
            "model.layers.3.self_attn.o_proj.weight","model.layers.1.linear_attn.out_proj.weight","down_proj"] {
            assert!(second_matrix(name),"{name}");assert_eq!(nv_role(name),NV_SECOND);
        }
        for name in ["model.layers.3.mlp.experts.7.gate_proj.weight","model.layers.3.mlp.experts.7.up_proj.weight","model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.kv_b_proj.weight","model.layers.3.mlp.gate.weight","model.layers.3.down_proj.extra.weight","lm_head.weight","o_proj_norm.weight"] {
            assert!(!second_matrix(name),"{name}");assert_eq!(nv_role(name),NV_FIRST);
        }
    }
    fn mixed(index:usize,salt:usize)->u32 {
        let mut value=(index as u32).wrapping_add((salt as u32+1).wrapping_mul(0x9e37_79b9));
        value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
    }
    /// Review finding F2 of the tensor-core rebase: the host fallback gives the contract word of the independent twin,
    /// for both roles, and NOT the rc.8 chain it gave before. The last BF16 rounding hides most order differences
    /// (about one word in a thousand moves on random inputs), so the test searches seeded rows for words that tell
    /// the three functions apart and checks the fallback on exactly those rows, and on plain rows beside them.
    #[test]
    fn the_host_fallback_gives_the_contract_word_of_the_role() {
        let cols=512usize;let (global_scale,input_scale)=(0.8125f32,0.4375f32);
        let row_of=|seed:usize|->(Vec<u8>,Vec<u8>,Vec<f32>) {
            let weight=(0..cols/2).map(|i|mixed(i,7*seed+1) as u8).collect();
            let scales=(0..cols/16).map(|i|if mixed(i,7*seed+2).is_multiple_of(19) {0} else {0x18+(mixed(i,7*seed+3)%96) as u8}).collect();
            let input=(0..cols).map(|i|match mixed(i,7*seed+4)%11 {0=>0.0,1=>-0.0,_=>((mixed(i,7*seed+5)%4001) as f32-2000.0)/(1u32<<(mixed(i,7*seed+6)%9)) as f32/37.0}).collect();
            (weight,scales,input)
        };
        // The three words of one row: first role, second role (both from the twin), and the rc.8 chain of the old fallback.
        let words=|weight:&[u8],scales:&[u8],input:&[f32]|->[u32;3] {
            let activation=Activation::from_bf16_source(input,input_scale).unwrap();
            let a:Vec<u8>=(0..cols).map(|c|(activation.codes()[c/2]>>((c%2)*4))&15).collect();
            let codes:Vec<u8>=weight.iter().flat_map(|b|[b&15,b>>4]).collect();
            let twin=|chains|tc_twin::word(tc_twin::accumulator(chains,&codes,&a,scales,activation.scales()),global_scale,activation.global());
            let combined=global_scale*activation.global();let mut sum=0.0f32;
            for (block,&code) in scales.iter().enumerate().take(cols/16) {let scale=(nvfp4::e4m3(code)*nvfp4::e4m3(activation.scales()[block]))*combined;
                for c in block*16..block*16+16 {sum+=(nvfp4::e2m1(a[c])*nvfp4::e2m1(codes[c]))*scale;}}
            [twin(1),twin(4),expert_ops::round(sum).to_bits()]
        };
        let (mut by_role,mut by_age,mut plain)=(Vec::new(),Vec::new(),Vec::new());
        for seed in 0..200_000usize {
            let (weight,scales,input)=row_of(seed);let w=words(&weight,&scales,&input);
            if w[0]!=w[1] && by_role.len()<3 {by_role.push(seed);}
            if w[2]!=w[0] && by_age.len()<3 {by_age.push(seed);}
            if plain.len()<16 {plain.push(seed);}
            if by_role.len()==3 && by_age.len()==3 {break;}
        }
        assert!(!by_role.is_empty(),"no row tells the first role from the second");assert!(!by_age.is_empty(),"no row tells the contract word from the rc.8 chain");
        for seed in by_role.iter().chain(&by_age).chain(&plain).copied() {
            let (weight,scales,input)=row_of(seed);let w=words(&weight,&scales,&input);
            for (role,want) in [(NV_FIRST,w[0]),(NV_SECOND,w[1])] {
                let got=nv_host_product(1,cols,&weight,&scales,global_scale,input_scale,role,&input).unwrap();
                assert_eq!((got.len(),got[0].to_bits()),(1,want),"seed={seed} role={role}");
            }
        }
        // Several rows in one call: each row is its own word.
        let seeds=[by_role[0],by_age[0],plain[1]];let rows:Vec<_>=seeds.iter().map(|&s|row_of(s)).collect();
        let weight:Vec<u8>=rows.iter().flat_map(|r|r.0.clone()).collect();let scales:Vec<u8>=rows.iter().flat_map(|r|r.1.clone()).collect();
        let got=nv_host_product(3,cols,&weight,&scales,global_scale,input_scale,NV_SECOND,&rows[0].2).unwrap();
        for (row,word) in got.iter().enumerate() {assert_eq!(word.to_bits(),words(&rows[row].0,&rows[row].1,&rows[0].2)[1],"row={row}");}
        let (weight,scales,input)=row_of(0);
        assert!(nv_host_product(1,cols,&weight,&scales,global_scale,input_scale,NV_FIRST,&input[..cols-16]).is_err());
        // A second matrix needs whole quarters of 64-column steps.
        assert!(nv_host_product(1,64,&weight[..32],&scales[..4],global_scale,input_scale,NV_SECOND,&input[..64]).is_err());
    }
}
