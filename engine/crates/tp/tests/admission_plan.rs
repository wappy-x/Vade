use engine_model::{plan::{layer::Topology,state::StatePlan},safetensors::Dtype};
use spark_tp::{admission_plan::{SessionChargePlan,StateChargePlan},bounded_decoder::expanded_bytes,service_memory};
#[path="support/attention_checkpoint.rs"] mod fixture;

#[test]
fn cached_request_charges_equal_rebuilt_plans_at_pool_and_prefix_boundaries() {
    let _constructor:fn(Dtype)->fixture::Checkpoint=fixture::Checkpoint::new;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let cached=SessionChargePlan::new(&c.admission,70000).unwrap();
    let state=|n|StatePlan::new(c.admission.description(),Topology {rank:0,parts:1},n as u64,isize::MAX as usize).unwrap().state_layout().bytes();
    let mut positions=(1..=257).collect::<Vec<_>>();positions.extend([511,512,513,1023,1024,1025,1151,1152,1153,4158,8192,16384,32767,32768,32769,65535,65536,65544,70000]);
    for tokens in positions {
        assert_eq!(cached.state_bytes(tokens).unwrap(),state(tokens),"state tokens={tokens}");
        for prompt in [0,1,tokens/2,tokens.saturating_sub(64),tokens] {
            let rebuilt=service_memory::session(&c.admission,prompt,tokens).unwrap();
            assert_eq!(cached.session(prompt,tokens).unwrap(),rebuilt,"session prompt={prompt} tokens={tokens}");
            let legacy=2*state(tokens)+2*expanded_bytes(&c.admission,prompt)+c.admission.description().vocab as usize*16;
            let slack=cached.append_reservation_bytes();
            assert_eq!(cached.request(prompt,tokens).unwrap(),legacy.max(rebuilt-slack)+slack,"request prompt={prompt} tokens={tokens}");
        }
    }
    assert!(cached.session(2,1).is_err());assert!(cached.session(0,0).is_err());assert!(cached.request(1,70001).is_err());
}

#[test]
fn cache_capacity_is_funded_once_per_bank_and_layer() {
    use engine_model::plan::description::TokenMixer;
    use spark_tp::admission_plan::cache_append_reservation_bytes;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let mut description=c.admission.description().clone();
    let latent=description.layers.iter().find(|layer|matches!(layer.mixer,TokenMixer::LatentAttention(_))).unwrap().clone();
    description.layers=vec![latent;11];
    for layer in &mut description.layers {let TokenMixer::LatentAttention(a)=&mut layer.mixer else {unreachable!()};a.kv_rank=512;}
    // Independent literal oracle: 11 layers, 128 spare rows, 512 words, 4 bytes,
    // two simultaneous banks. Already page aligned. No production helper in the oracle.
    assert_eq!(cache_append_reservation_bytes(&description).unwrap(),5_767_168);
    description.layers.push(description.layers[0].clone());
    assert_eq!(cache_append_reservation_bytes(&description).unwrap(),6_291_456);
    description.layers.truncate(1);
    let TokenMixer::LatentAttention(a)=&mut description.layers[0].mixer else {unreachable!()};a.kv_rank=1;
    assert_eq!(cache_append_reservation_bytes(&description).unwrap(),8192,"both banks include page rounding");
    description.layers.clear();assert_eq!(cache_append_reservation_bytes(&description).unwrap(),0);
    let cached=SessionChargePlan::new(&c.admission,70000).unwrap();
    let slack=cached.append_reservation_bytes();assert!(slack>cached.cache_append_reservation_bytes());
    for (prompt,tokens) in [(1,2),(64,128),(4158,8192),(16384,32768)] {
        let state=StatePlan::new(c.admission.description(),Topology {rank:0,parts:1},tokens,isize::MAX as usize).unwrap().state_layout().bytes();
        let legacy=2*state+2*expanded_bytes(&c.admission,prompt)+c.admission.description().vocab as usize*16;
        assert!(cached.request(prompt,tokens as usize).unwrap()>=legacy+slack);
    }
}

#[test]
fn index_cache_capacity_is_funded_once_per_bank_cache_and_layer() {
    use engine_model::plan::description::TokenMixer;
    use spark_tp::admission_plan::{append_reservation_bytes,cache_append_reservation_bytes,index_append_reservation_bytes};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let mut description=c.admission.description().clone();
    let latent=description.layers.iter().find(|layer|matches!(layer.mixer,TokenMixer::LatentAttention(_))).unwrap().clone();
    description.layers=vec![latent;11];
    for layer in &mut description.layers {let TokenMixer::LatentAttention(a)=&mut layer.mixer else {unreachable!()};a.kv_rank=512;a.indexer.dim=128;}
    // Independent literal oracle: 11 layers, keys and gates, two banks, 128 spare rows,
    // 128 words, 4 bytes. Already page aligned. No production helper in the oracle.
    assert_eq!(index_append_reservation_bytes(&description).unwrap(),2_883_584);
    assert_eq!(append_reservation_bytes(&description).unwrap(),8_650_752,"latent and index terms add, no global rounding");
    assert_eq!(cache_append_reservation_bytes(&description).unwrap(),5_767_168,"the latent term is unchanged");
    description.layers.push(description.layers[0].clone());
    assert_eq!(index_append_reservation_bytes(&description).unwrap(),3_145_728);
    description.layers.truncate(1);
    let TokenMixer::LatentAttention(a)=&mut description.layers[0].mixer else {unreachable!()};a.indexer.dim=1;
    assert_eq!(index_append_reservation_bytes(&description).unwrap(),16_384,"each cache and bank rounds to its own page");
    description.layers.clear();assert_eq!(index_append_reservation_bytes(&description).unwrap(),0);
    let cached=SessionChargePlan::new(&c.admission,70000).unwrap();
    assert!(cached.index_append_reservation_bytes()>0);
    assert_eq!(cached.append_reservation_bytes(),cached.cache_append_reservation_bytes()+cached.index_append_reservation_bytes());
    for (prompt,tokens) in [(1,2),(64,128),(4158,8192)] {
        assert_eq!(cached.session(prompt,tokens).unwrap(),service_memory::session(&c.admission,prompt,tokens).unwrap(),"one charge, one term");
    }
}

#[test]
fn cached_predictor_state_equals_rebuilt_plan() {
    let c=fixture::Checkpoint::new_with_predictor(70000);let _=&c.operands;
    let mut description=c.admission.description().with_draft_layer().unwrap().unwrap();description.residual=c.admission.description().residual;
    let cached=StateChargePlan::new(&description,70000).unwrap();
    for n in [1,2,3,7,8,9,127,128,129,255,256,257,1151,1152,1153,32768,65544,70000] {
        let rebuilt=StatePlan::new(&description,Topology {rank:0,parts:1},n as u64,isize::MAX as usize).unwrap();
        assert_eq!(cached.bytes(n).unwrap(),rebuilt.state_layout().bytes(),"predictor tokens={n}");
    }
}

#[test]
fn cached_admission_cpu_receipt() {
    use std::{hint::black_box,time::Instant};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,70000);let _=&c.operands;
    let cached=SessionChargePlan::new(&c.admission,32768).unwrap();
    let start=Instant::now();for _ in 0..20 {black_box(service_memory::session(&c.admission,4158,12350).unwrap());}let rebuilt=start.elapsed();
    let start=Instant::now();for _ in 0..20000 {black_box(cached.session(black_box(4158),black_box(12350)).unwrap());}let reused=start.elapsed();
    eprintln!("ADMISSION CPU synthetic layers={} rebuilt_ns_per_call={} cached_ns_per_call={} (startup excluded)",c.admission.description().layers.len(),rebuilt.as_nanos()/20,reused.as_nanos()/20000);
}
