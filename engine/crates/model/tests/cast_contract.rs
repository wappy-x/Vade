use engine_model::plan::{ConfigPlan, layer::{*,cast::*}};
use std::collections::BTreeMap;

fn description() -> engine_model::plan::TextDescription {
    let ConfigPlan::Deferred(d)=ConfigPlan::from_config(include_bytes!("fixtures/glm/config.json")).unwrap() else { panic!() }; d
}
fn bits(x: &[f32]) -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() }

#[test]
fn finite_bf16_values_round_trip_including_signed_zero_and_subnormals() {
    // Exhaust the finite BF16 encoding space. Expected values come from bits.
    for raw in 0..=u16::MAX {
        if raw & 0x7f80==0x7f80 { continue; }
        let value=f32::from_bits((raw as u32)<<16);
        assert_eq!(round_bf16(value).unwrap().to_bits(),value.to_bits());
    }
}
#[test]
fn nearest_even_ties_carry_and_finite_overflow_are_checked() {
    for (input,expected) in [(0x3f80_8000,0x3f80_0000),(0x3f81_8000,0x3f82_0000),
        (0xbf80_8000,0xbf80_0000),(0xbf81_8000,0xbf82_0000),
        (0x0000_8000,0),(0x0001_8000,0x0002_0000),(0x8000_8000,0x8000_0000),
        (0x3fff_8000,0x4000_0000),(0x007f_8000,0x0080_0000),
        (0x7f7f_7fff,0x7f7f_0000)] {
        assert_eq!(round_bf16(f32::from_bits(input)).unwrap().to_bits(),expected);
    }
    for input in [0x7f7f_8000,0xff7f_8000,0x7f7f_ffff,0xff7f_ffff,0x7f80_0000,0xff80_0000,0x7fc0_0000] {
        assert!(round_bf16(f32::from_bits(input)).is_err());
    }
}
#[test]
fn ordinary_norm_casts_before_weight_multiply() {
    // Hand vector: normalized is the midpoint above 1; the BF16 weight is 1.5.
    // Casting only after the product gives the next BF16 value above 1.5.
    let normalized=1.00390625;
    assert_eq!(norm_weight(normalized,1.5).unwrap().to_bits(),1.5f32.to_bits());
    assert_eq!(round_bf16(normalized*1.5).unwrap().to_bits(),1.5078125f32.to_bits());
    assert!(norm_weight(1.0,1.001).is_err());
    assert!(norm_weight(f32::INFINITY,1.0).is_err());
    assert!(norm_weight(f32::from_bits(0x7f7f0000),2.0).is_err());
}
#[test]
fn expert_fold_rounds_each_add_and_uses_expert_id_order() {
    let ids=[0,1,2]; let weights=[1.0;3]; let down=[256.0,1.0,-256.0];
    // BF16 256+1 is a tie that rounds to 256. A final-only cast gives 1.
    assert_eq!(expert_fold(&ids,&weights,&down,&[0.0]).unwrap(),[0.0]);
    assert_eq!(expert_fold(&[2,1,0],&weights,&down,&[0.0]).unwrap(),[1.0]);
    // Slot permutation must not change the expert-ID fold.
    assert_eq!(expert_fold(&[2,0,1],&weights,&[-256.0,256.0,1.0],&[0.0]).unwrap(),[0.0]);
    // Shared is added last. It has no route weight.
    assert_eq!(expert_fold(&ids,&weights,&down,&[1.0]).unwrap(),[1.0]);
    // Route weights retain FP32 bits until the contribution cast.
    assert_eq!(expert_fold(&[7],&[1.00390625],&[1.5],&[0.0]).unwrap(),[1.5078125]);
    assert_eq!(expert_fold(&[7],&[1.00390625],&[1.5],&[0.0078125]).unwrap(),[1.515625]);
}
#[test]
fn expert_fold_is_per_column_and_rejects_invalid_inputs() {
    assert_eq!(expert_fold(&[5,2],&[0.5,2.0],&[2.0,-4.0,3.0,4.0],&[1.0,-1.0]).unwrap(),[8.0,5.0]);
    for (ids,weights,down,shared) in [
        (vec![],vec![],vec![],vec![0.0]),
        (vec![0],vec![],vec![1.0],vec![0.0]),
        (vec![0],vec![1.0],vec![1.0,2.0],vec![0.0]),
        (vec![0,0],vec![1.0,1.0],vec![1.0,1.0],vec![0.0]),
        (vec![u32::MAX],vec![1.0],vec![1.0],vec![0.0]),
        (vec![0],vec![f32::NAN],vec![1.0],vec![0.0]),
        (vec![0],vec![1.0],vec![1.001],vec![0.0]),
        (vec![0],vec![1.0],vec![1.0],vec![f32::INFINITY]),
        (vec![0],vec![f32::MAX],vec![2.0],vec![0.0]),
        (vec![0,1],vec![1.0,1.0],vec![f32::from_bits(0x7f7f0000);2],vec![0.0]),
    ] { assert!(expert_fold(&ids,&weights,&down,&shared).is_err()); }
}
#[test]
fn counted_gather_preserves_bits_and_never_reads_padding() {
    // Invalid unselected cache rows and arbitrary padded IDs must stay unread.
    let cache=[-0.0,f32::from_bits(0x00010000),f32::NAN,f32::INFINITY,3.0,-4.0];
    let output=gather_bf16(&cache,2,&[0,2,u32::MAX,1234],2,2).unwrap();
    assert_eq!(bits(&output),[0x80000000,0x00010000,3.0f32.to_bits(),(-4.0f32).to_bits(),0,0,0,0]);
    assert_eq!(bits(&gather_bf16(&cache,2,&[u32::MAX;3],0,2).unwrap()),[0;6]);
    assert_eq!(gather_bf16(&cache,2,&[],0,2).unwrap(),[]);
    for (width,ids,count,pos) in [(0,vec![0],1,2),(4,vec![0],1,0),(2,vec![0],2,2),
        (2,vec![0],1,3),(2,vec![2],1,1),(2,vec![0,0],2,2),(2,vec![2,0],2,2),
        (2,vec![u32::MAX],1,2),(2,vec![1],1,2)] {
        assert!(gather_bf16(&cache,width,&ids,count,pos).is_err());
    }
    assert!(gather_bf16(&[1.001],1,&[0],1,0).is_err());
}
#[test]
fn every_decoder_boundary_has_a_source_and_stays_refused() {
    let d=description();
    for layer in 0..d.layers.len() {
        let p=LayerPlan::new(&d,layer,7,Topology {parts:1,rank:0}).unwrap();
        let c=p.bf16_cast_contract().unwrap();
        assert!(c.require_complete().is_err());
        assert_eq!(c.steps.len(),p.steps.len());
        let mut known=c.inputs.iter().map(|v| (v.buffer.name.clone(),v.clone())).collect::<BTreeMap<_,_>>();
        for (boundary,step) in c.steps.iter().zip(&p.steps) {
            assert_eq!((&boundary.id,boundary.kind),(&step.id,step.kind));
            assert_eq!(boundary.weights.iter().map(|w| &w.weight).collect::<Vec<_>>(),step.weights.iter().collect::<Vec<_>>());
            assert!(!boundary.source.is_empty() && !boundary.rounding.is_empty() && !boundary.primitive_order.is_empty() && !boundary.refused.is_empty());
            for (input,name) in boundary.inputs.iter().zip(&step.inputs) { assert_eq!(input,&known[name]); }
            for output in &boundary.outputs { known.insert(output.buffer.name.clone(),output.clone()); }
        }
        for output in c.returns { assert_eq!(output,known[&output.buffer.name]); }
    }
    assert!(ConfigPlan::Deferred(d).into_model_plan().is_err());
}
#[test]
fn contract_separates_state_output_coefficients_and_router_types() {
    let d=description(); let mut saw_kda=false; let mut saw_latent=false; let mut saw_route=false;
    for layer in 0..d.layers.len() {
        let p=LayerPlan::new(&d,layer,7,Topology {parts:1,rank:0}).unwrap();
        let c=p.bf16_cast_contract().unwrap();
        for s in &c.steps {
            match s.kind {
                Kind::MhcPrepare => { assert_eq!(s.outputs[0].dtype,Dtype::F32); assert_eq!(s.scratch[0].dtype,Dtype::F32); },
                Kind::Norm => { assert_eq!(s.accumulator,Dtype::F32); assert_eq!(s.outputs[0].dtype,Dtype::Bf16); },
                Kind::KdaStep => { saw_kda=true; assert_eq!(s.inputs[1].dtype,Dtype::F32);
                    assert_eq!(s.outputs.iter().map(|p| p.dtype).collect::<Vec<_>>(),[Dtype::Bf16,Dtype::F32]); },
                Kind::CacheAppend => { saw_latent=true; assert_eq!(s.inputs[0].dtype,Dtype::Unresolved);
                    assert_eq!(s.inputs[1].dtype,Dtype::Bf16); assert_eq!(s.outputs[0].dtype,Dtype::Unresolved); },
                Kind::Route => { saw_route=true; assert_eq!(s.inputs[0].dtype,Dtype::F32);
                    assert_eq!(s.outputs.iter().map(|p| p.dtype).collect::<Vec<_>>(),[Dtype::U32,Dtype::F32]); },
                Kind::Softmax => { assert_eq!(s.accumulator,Dtype::F32); assert_eq!(s.outputs[0].dtype,Dtype::Bf16); },
                _=>(),
            }
        }
    }
    assert!(saw_kda && saw_latent && saw_route);
}
#[test]
fn contract_uses_operation_data_and_is_independent_of_topology() {
    let d=description();
    for layer in [0,3] {
        let base=LayerPlan::new(&d,layer,7,Topology {parts:1,rank:0}).unwrap().bf16_cast_contract().unwrap();
        for parts in [2,4] { for rank in 0..parts {
            let split=LayerPlan::new(&d,layer,7,Topology {parts,rank}).unwrap().bf16_cast_contract().unwrap();
            assert_eq!(base,split);
        } }
    }
    let mut plain=d.clone(); plain.storage.layers.clear(); plain.storage.excluded.clear();
    let p=LayerPlan::new(&plain,3,7,Topology {parts:1,rank:0}).unwrap();
    let c=p.bf16_cast_contract().unwrap();
    let router=c.steps.iter().find(|s| s.outputs.iter().any(|o| o.buffer.name=="router_logits")).unwrap();
    assert_eq!((router.weights[0].parameter,router.weights[0].operand),(Dtype::Bf16,Dtype::F32));
    let mut p=LayerPlan::new(&d,0,7,Topology {parts:1,rank:0}).unwrap();
    p.steps[0].inputs[0]="absent".into();
    assert!(p.bf16_cast_contract().is_err());
}
