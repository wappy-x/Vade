use spark_tp::{bounded_decoder::{Decoder,Host},causal_layer::Layer};
use engine_model::safetensors::Dtype;
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn bounded_stack_keeps_the_proved_dense_layer_bytes_and_commits_all_layers() {
    let old=fixture::Checkpoint::new(Dtype::Bf16);assert!(!old.operands.is_empty());
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);assert!(!c.operands.is_empty());
    let decoder=Decoder::new(&c.admission,Host,4).unwrap();let zero=decoder.zero_state().unwrap();
    let residual=vec![0.25;c.admission.description().hidden as usize*c.admission.description().residual.streams as usize];
    let layer=Layer::new(&c.admission,0).unwrap();let (expected,state)=layer.reference(&residual,&layer.zero_state().unwrap()).unwrap();
    let (actual,next)=decoder.layer(0,1,&residual,&zero.layers[0]).unwrap();assert_eq!(actual,expected["output"]);
    assert_eq!(next["kda.history"],state.history);assert_eq!(next["kda.state"],state.recurrent);
    let (logits,id,first)=decoder.step(1,&zero,|_,_|Ok(())).unwrap();assert_eq!(first.position,1);assert_eq!(zero.position,0);
    assert_eq!(logits.len(),c.admission.description().vocab as usize);assert!((id as u64)<c.admission.description().vocab);
    let repeat=decoder.step(1,&zero,|_,_|Ok(())).unwrap();assert_eq!(logits,repeat.0);assert_eq!(id,repeat.1);
    let (_,_,second)=decoder.step(id,&first,|_,_|Ok(())).unwrap();assert_eq!(second.position,2);
    assert!(decoder.step(1,&first,|_,_|Err("observer failure".into())).is_err());assert_eq!(first.position,1);
}
#[test]
fn nvfp4_operation_matches_the_calibrated_projection_reference() {
    use engine_model::flash::{activation_nvfp4::Activation,nvfp4::{e2m1,e4m3}};
    use spark_gpu_probe::causal_ops::{reference,Op,Shape};
    let (rows,cols)=(3,32);
    let input=(0..cols).map(|i|(i as f32-17.0)*0.375).collect::<Vec<_>>();
    let weight=(0..rows*cols/2).map(|i|((i*13+7)%256) as u8).collect::<Vec<_>>();
    let scales=[0x30,0x38,0x40,0x28,0x42,0x34];
    for (activation_global,weight_global) in [(0.125,0.75),(1.5,2.25),(0.03125,0.0625)] {
        let a=Activation::from_bf16_source(&input,activation_global).unwrap();
        let expected=a.project(&weight,&scales,weight_global).unwrap();
        let values=(0..cols).map(|i|e2m1(a.codes()[i/2]>>((i%2)*4))).collect::<Vec<_>>();
        let weights=(0..rows*cols).map(|i|e2m1(weight[i/2]>>((i%2)*4))).collect::<Vec<_>>();
        let combined=weight_global*a.global();
        let blocks=scales.iter().enumerate().map(|(i,&s)|(e4m3(s)*e4m3(a.scales()[i%(cols/16)]))*combined).collect::<Vec<_>>();
        let shape=Shape {op:Op::Nvfp4,heads:0,dim:0,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
        let actual=reference(shape,[&values,&blocks,&[],&[]],&weights).unwrap();
        assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        let mut filled=vec![f32::NAN;rows];
        for (i,v) in filled.iter_mut().enumerate() {spark_gpu_probe::causal_ops::unit(shape,i,[&values,&blocks,&[],&[]],&weights,std::slice::from_mut(v));}
        assert_eq!(filled,actual);
    }
}
#[test]
fn prompt_groups_keep_every_logit_and_recurrent_cache_bit_across_tile_tails() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let decoder=Decoder::new(&c.admission,Host,8).unwrap();let tokens=[1,7,2,5,3,6,4];
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    for width in 1..=spark_tp::bounded_decoder::PROMPT_ROWS {
        let mut scalar=decoder.zero_state().unwrap();let mut grouped=decoder.zero_state().unwrap();
        for chunk in tokens.chunks(width) {
            let mut expected=Vec::new();for &token in chunk {let (logits,_,next)=decoder.step(token,&scalar,|_,_|Ok(())).unwrap();scalar=next;expected.push(logits);}
            let (actual,next)=decoder.prefill(chunk,&grouped).unwrap();grouped=next;
            assert_eq!(actual.iter().map(|v|bits(v)).collect::<Vec<_>>(),expected.iter().map(|v|bits(v)).collect::<Vec<_>>());
            assert_eq!(grouped.position,scalar.position);
            for (a,b) in grouped.layers.iter().zip(&scalar.layers) {assert_eq!(a.keys().collect::<Vec<_>>(),b.keys().collect::<Vec<_>>());for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"width={width} state={name}");}}
        }
    }
}
#[test]
fn causal_attention_prefix_keeps_each_live_fold_and_removes_only_unused_padding() {
    use spark_gpu_probe::causal_ops::{reference,Op,Shape};
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    for tail in [false,true] {for context in [1usize,2,3,4,5,6,7,8,9,12] {
        let (heads,key,value,latent,pool,budget)=(2usize,3usize,2usize,4usize,3usize,6usize);
        let slots=budget+if tail {pool-1} else {0};let compact=slots.min(context);
        let shape=|op|Shape {op,heads:heads as u32,dim:key as u32,taps:0,rows:slots as u32,cols:0,epsilon:1e-5,bound:0.0};
        let select=Shape {rows:context as u32,dim:pool as u32,taps:budget as u32,cols:u32::from(tail),..shape(Op::Select)};
        let pool_scores=(0..context/pool).map(|i|(i%3) as f32*0.03125).collect::<Vec<_>>();
        let selected=reference(select,[&pool_scores,&[],&[],&[]],&[]).unwrap();let count=[selected[slots]];let live=count[0] as usize;
        assert!(live<=compact,"context={context} tail={tail}");
        let cache=(0..context*latent).map(|i|if i%7==0 {-0.0} else {(i as i32%17-8) as f32*0.037}).collect::<Vec<_>>();
        let gather=Shape {rows:context as u32,dim:latent as u32,cols:slots as u32,..shape(Op::Gather)};
        let full=reference(gather,[&cache,&selected[..slots],&count,&[]],&[]).unwrap();
        let small=reference(Shape {cols:compact as u32,..gather},[&cache,&selected[..compact],&count,&[]],&[]).unwrap();
        assert_eq!(bits(&full[..small.len()]),bits(&small));
        // Identical expanded live rows stand in for the unchanged projection.
        let packed=(0..live*heads*(key+value)).map(|i|if i%11==0 {-0.0} else {(i as i32%23-11) as f32*0.019}).collect::<Vec<_>>();
        let unpack=Shape {cols:value as u32,taps:live as u32,..shape(Op::UnpackKv)};
        let full=reference(unpack,[&packed,&[],&[],&[]],&[]).unwrap();
        let small=reference(Shape {rows:compact as u32,..unpack},[&packed,&[],&[],&[]],&[]).unwrap();
        let (full_keys,full_values)=full.split_at(slots*heads*key);let (small_keys,small_values)=small.split_at(compact*heads*key);
        assert_eq!(bits(&full_keys[..small_keys.len()]),bits(small_keys));
        assert_eq!(bits(&full_values[..small_values.len()]),bits(small_values));
        let query=(0..heads*key).map(|i|(i as f32-3.0)*0.23).collect::<Vec<_>>();
        let score=shape(Op::Scores);let full_scores=reference(score,[&query,full_keys,&count,&[]],&[]).unwrap();
        let small_scores=reference(Shape {rows:compact as u32,..score},[&query,small_keys,&count,&[]],&[]).unwrap();
        let softmax=shape(Op::Softmax);let full_prob=reference(softmax,[&full_scores,&count,&[],&[]],&[]).unwrap();
        let small_prob=reference(Shape {rows:compact as u32,..softmax},[&small_scores,&count,&[],&[]],&[]).unwrap();
        for head in 0..heads {
            assert_eq!(bits(&full_scores[head*slots..head*slots+compact]),bits(&small_scores[head*compact..(head+1)*compact]));
            assert_eq!(bits(&full_prob[head*slots..head*slots+compact]),bits(&small_prob[head*compact..(head+1)*compact]));
        }
        let fold=Shape {dim:value as u32,..shape(Op::Values)};
        let full=reference(fold,[&full_prob,full_values,&count,&[]],&[]).unwrap();
        let small=reference(Shape {rows:compact as u32,..fold},[&small_prob,small_values,&count,&[]],&[]).unwrap();
        assert_eq!(bits(&full),bits(&small),"context={context} tail={tail} live={live}");
    }}
}
#[test]
fn recurrent_schedule_geometry_is_position_invariant_for_dense_and_expert_layers() {
    use engine_model::plan::{layer::{LayerPlan,Topology},description::TokenMixer};
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let mut d=c.admission.description().clone();d.max_positions=320;
    let expert=d.layers[1].ffn;
    for ffn in [d.layers[0].ffn,expert] {d.layers[0].ffn=ffn;
        assert!(matches!(d.layers[0].mixer,TokenMixer::Kda(_)));
        for parts in [1,2,4] {for rank in 0..parts {let topology=Topology {rank,parts};let first=LayerPlan::new(&d,0,1,topology).unwrap();
            for context in 2..=320 {let mut later=LayerPlan::new(&d,0,context,topology).unwrap();later.context=1;assert_eq!(later,first);}
        }}
    }
    let first=LayerPlan::new(&d,1,1,Topology {rank:0,parts:1}).unwrap();
    let mut later=LayerPlan::new(&d,1,2,Topology {rank:0,parts:1}).unwrap();later.context=1;
    assert_ne!(later,first,"latent cache geometry must remain position dependent");
}
