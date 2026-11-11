use spark_tp::{bounded_decoder::{Decoder,Host},causal_layer::Layer};
use engine_model::safetensors::Dtype;
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn expanded_owner_refuses_a_different_topology_before_reuse() {
    use spark_tp::{bounded_decoder::HeadRows,sparse_heads::Layout};
    use engine_model::plan::layer::Topology;
    let checkpoint=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let decoder=Decoder::new(&checkpoint.admission,Host,8).unwrap();
    let (_,mut state)=decoder.prefill_rows(&[1,2],&decoder.zero_state().unwrap(),HeadRows::All).unwrap();
    let owner=state.expanded.values_mut().next().unwrap();let layout=owner.2;
    assert_eq!(owner.clone().2,layout);
    owner.2=Layout::new(layout.total_heads() as u64,layout.key_dim() as u64,layout.value_dim() as u64,Topology {rank:1,parts:2}).unwrap();
    assert!(decoder.step(3,&state,|_,_|Ok(())).err().unwrap().contains("layout differs"));
    assert!(decoder.prefill_rows(&[3,4],&state,HeadRows::All).err().unwrap().contains("layout differs"));
}
#[test]
fn step_allocation_fields_keep_the_verify_line_names() {
    use spark_tp::bounded_decoder::{allocation_fields,Compute};
    // RESIDENT TOKEN and SERVED STEP print these two fields under the names RESIDENT VERIFY uses.
    assert_eq!(allocation_fields(Some((30,1<<20)),Some((33,(1<<20)+156_237_824))),"new_buffers=3 new_bytes=156237824");
    assert_eq!(allocation_fields(Some((5,4096)),Some((5,4096))),"new_buffers=0 new_bytes=0");
    assert_eq!(allocation_fields(None,Some((1,1))),"new_buffers=none new_bytes=none");
    assert_eq!(allocation_fields(Some((1,1)),None),"new_buffers=none new_bytes=none");
    assert_eq!(Compute::vector_allocations(&Host),None,"the host reference has no device vectors");
    // Prompt groups add the pool's reused spares under the name RESIDENT VERIFY's line uses.
    use spark_tp::bounded_decoder::group_allocation_fields;
    assert_eq!(group_allocation_fields(Some((10,4096,7)),Some((12,12288,30))),"new_buffers=2 new_bytes=8192 reused_buffers=23");
    assert_eq!(group_allocation_fields(Some((1,1,1)),None),"new_buffers=none new_bytes=none reused_buffers=none");
    assert_eq!(Compute::vector_pool_counts(&Host),None);
}
#[test]
fn read_pair_default_keeps_both_logical_vectors_and_signed_zero_bits() {
    use spark_tp::bounded_decoder::Compute;
    let first=vec![0.0,-0.0,1.0];let second=vec![7.0,-3.0];
    let (a,b)=Compute::read_pair(&Host,&first,&second).unwrap();
    assert_eq!(a.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),first.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
    assert_eq!(b.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),second.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
}
#[test]
fn a_latent_projection_stored_as_f32_is_admitted_and_decodes() {
    // The device gate select-known uses this form for the expanded path: latent absorption admits BF16 storage only.
    let c=fixture::Checkpoint::build_typed(Dtype::Bf16,true,8,false,|_|{},|name|name.ends_with("kv_b_proj.weight").then_some(Dtype::F32));
    let typed=c.admission.headers().weights().iter().filter(|(name,view)|name.ends_with("kv_b_proj.weight") && view.components().first().map(|c|c.dtype)==Some(Dtype::F32)).count();
    assert!(typed>0,"no latent projection was stored as F32");
    let decoder=Decoder::new(&c.admission,Host,8).unwrap();let mut state=decoder.zero_state().unwrap();
    for token in [1,3,2,5] {let (logits,_,next)=decoder.step(token,&state,|_,_|Ok(())).unwrap();assert!(logits.iter().all(|v|v.is_finite()));state=next;}
}
#[test]
fn compact_greedy_step_keeps_ids_and_state_bytes_of_full_logits() {
    let checkpoint=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let decoder=Decoder::new(&checkpoint.admission,Host,8).unwrap();
    let mut full=decoder.zero_state().unwrap();let mut compact=decoder.zero_state().unwrap();let mut token=1;
    for _ in 0..8 {
        let (_,expected,next_full)=decoder.step(token,&full,|_,_|Ok(())).unwrap();
        let (actual,next_compact)=decoder.step_greedy(token,&compact,|_,_|Ok(())).unwrap();assert_eq!(actual,expected);
        assert_eq!(next_full.position,next_compact.position);
        for (full,compact) in next_full.layers.iter().zip(&next_compact.layers) {assert_eq!(full.keys().collect::<Vec<_>>(),compact.keys().collect::<Vec<_>>());
            for (name,values) in full {assert_eq!(values.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),compact[name].iter().map(|value|value.to_bits()).collect::<Vec<_>>());}
        }
        full=next_full;compact=next_compact;token=actual;
    }
    assert_eq!(compact.position,8);assert!(decoder.step_greedy(token,&compact,|_,_|Ok(())).is_err());assert_eq!(compact.position,8);
    let initial=decoder.zero_state().unwrap();assert!(decoder.step_greedy(1,&initial,|_,_|Err("observer failure".into())).is_err());assert_eq!(initial.position,0);
}
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
fn prepared_parameters_stay_bound_to_their_checkpoint() {
    // `from_prepared` takes no second checkpoint, so a mixed pair cannot be written.
    let (a,b)=(fixture::Checkpoint::new_with_head(Dtype::Bf16,true),fixture::Checkpoint::new_with_head(Dtype::F32,true));
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let first=|d:&Decoder<'_,Host>|bits(&d.step(1,&d.zero_state().unwrap(),|_,_|Ok(())).unwrap().0);
    let (fresh_a,fresh_b)=(Decoder::new(&a.admission,Host,8).unwrap(),Decoder::new(&b.admission,Host,8).unwrap());
    let reused=Decoder::from_prepared(&fresh_a.prepared(),Host,8,0).unwrap();
    assert_eq!(first(&reused),first(&fresh_a));
    assert_ne!(first(&reused),first(&fresh_b),"the two checkpoints must differ for this test to mean anything");
    assert!(Decoder::from_prepared(&fresh_a.prepared(),Host,9,0).is_err());
}
/// A service with the predictor keeps only the trail of a large prompt group: the last rows, with their bits, in storage of their own.
#[test]
fn the_open_tail_of_a_group_holds_its_last_rows_bits() {
    use spark_tp::bounded_decoder::HeadRows;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let decoder=Decoder::new(&c.admission,Host,80).unwrap();
    let tokens=(0..40usize).map(|i|((i*5+1)%16) as u32).collect::<Vec<_>>();
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let (_,_,full)=decoder.prefill_rows_hidden(&tokens,&decoder.zero_state().unwrap(),HeadRows::Last).unwrap();
    let (_,_,open)=decoder.prefill_rows_hidden(&tokens,&decoder.zero_state().unwrap(),HeadRows::Last).unwrap();
    let tail=decoder.open_tail(open,8).unwrap();
    for row in 0..8 {
        let [a,b]=decoder.open_hidden(&tail,row).unwrap();let [x,y]=decoder.open_hidden(&full,32+row).unwrap();
        assert_eq!((bits(&a),bits(&b)),(bits(&x),bits(&y)),"row {row}");
    }
    assert!(decoder.open_hidden(&tail,8).is_err(),"the tail holds eight rows");
    // A group no larger than the trail is returned as it is.
    let (_,_,small)=decoder.prefill_rows_hidden(&tokens[..8],&decoder.zero_state().unwrap(),HeadRows::Last).unwrap();
    let same=decoder.open_tail(small,8).unwrap();assert!(decoder.open_hidden(&same,7).is_ok() && decoder.open_hidden(&same,8).is_err());
}
/// Groups past 16 rows that follow earlier rows fold only the index pools they complete
/// (one window launch). Their last-row logits and every state plane have the bits of scalar steps.
#[test]
fn groups_past_sixteen_rows_fold_only_new_pools_and_keep_every_state_bit() {
    use spark_tp::bounded_decoder::HeadRows;
    let context=80u64;
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,context);
    let decoder=Decoder::new(&c.admission,Host,context).unwrap();
    let tokens=(0..70usize).map(|i|((i*7+3)%16) as u32).collect::<Vec<_>>();
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let windows=||spark_gpu_probe::prompt_launches::line().split(' ').find_map(|f|f.strip_prefix("pool_window=")?.parse::<u64>().ok()).unwrap();
    let before=windows();
    for (width,limit) in [(17usize,80usize),(20,80),(33,80),(20,4),(33,0)] {
        decoder.set_expanded_limit(limit);
        let mut scalar=decoder.zero_state().unwrap();let mut grouped=decoder.zero_state().unwrap();
        for chunk in tokens.chunks(width) {
            let mut expected=Vec::new();for &token in chunk {let (logits,_,next)=decoder.step(token,&scalar,|_,_|Ok(())).unwrap();scalar=next;expected=logits;}
            let (mut actual,next)=decoder.prefill_rows(chunk,&grouped,HeadRows::Last).unwrap();grouped=next;
            assert_eq!(bits(&actual.pop().unwrap()),bits(&expected),"last row width={width} limit={limit}");
            assert_eq!(grouped.position,scalar.position);
            for (a,b) in grouped.layers.iter().zip(&scalar.layers) {assert_eq!(a.keys().collect::<Vec<_>>(),b.keys().collect::<Vec<_>>());for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"{name} width={width} limit={limit}");}}
        }
    }
    assert!(windows()>before,"no group took the window fold");
}
/// A large group that ends inside a pool hands its unpooled rows to decode steps. 70 tokens as [50, 20] with pools of 4 leave
/// rows 68 and 69 outside a complete pool after the window fold, and the steps after it complete that pool. With pools of 32 a
/// 17-row group completes no pool at all in its window fold, and a large group that starts inside the first pool takes the full
/// fold. In every case each last-row logit, each state plane and each decode step after the prompt has the bits of scalar steps.
#[test]
fn a_window_fold_tail_hands_unpooled_rows_to_decode_steps() {
    use spark_tp::bounded_decoder::HeadRows;
    use engine_model::plan::description::{TextDescription,TokenMixer};
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let windows=||spark_gpu_probe::prompt_launches::line().split(' ').find_map(|f|f.strip_prefix("pool_window=")?.parse::<u64>().ok()).unwrap();
    for (pool,budget,splits) in [(4u64,16u64,vec![50usize,20]),(32,64,vec![33,17]),(32,64,vec![40,17,13]),(32,64,vec![10,30,20])] {
        let context=80u64;
        let c=fixture::Checkpoint::build_with(Dtype::Bf16,true,context,false,|d:&mut TextDescription| {
            for layer in d.layers.iter_mut() {if let TokenMixer::LatentAttention(a)=&mut layer.mixer {a.indexer.pool=pool;a.indexer.token_budget=budget;}}
        });
        let decoder=Decoder::new(&c.admission,Host,context).unwrap();
        let tokens=(0..splits.iter().sum::<usize>()).map(|i|((i*7+3)%16) as u32).collect::<Vec<_>>();
        let before=windows();
        let mut scalar=decoder.zero_state().unwrap();let mut grouped=decoder.zero_state().unwrap();let mut at=0;
        for &m in &splits {
            let chunk=&tokens[at..at+m];at+=m;
            let mut expected=Vec::new();for &token in chunk {let (logits,_,next)=decoder.step(token,&scalar,|_,_|Ok(())).unwrap();scalar=next;expected=logits;}
            let (mut actual,next)=decoder.prefill_rows(chunk,&grouped,HeadRows::Last).unwrap();grouped=next;
            assert_eq!(bits(&actual.pop().unwrap()),bits(&expected),"last row pool={pool} splits={splits:?} rows={m}");
            assert_eq!(grouped.position,scalar.position);
            for (a,b) in grouped.layers.iter().zip(&scalar.layers) {for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"{name} pool={pool} splits={splits:?} rows={m}");}}
        }
        assert!(windows()>before,"no group took the window fold: pool={pool} splits={splits:?}");
        // Decode from the state the groups left and from the state scalar steps left: the same bits, step by step.
        let mut token=5u32;
        for step in 0..4 {
            let (a,best_a,next_a)=decoder.step(token,&grouped,|_,_|Ok(())).unwrap();let (b,best_b,next_b)=decoder.step(token,&scalar,|_,_|Ok(())).unwrap();
            assert_eq!(bits(&a),bits(&b),"decode step {step} pool={pool} splits={splits:?}");assert_eq!(best_a,best_b);
            for (x,y) in next_a.layers.iter().zip(&next_b.layers) {
                assert_eq!(x.keys().collect::<Vec<_>>(),y.keys().collect::<Vec<_>>());
                for (name,value) in x {assert_eq!(bits(value),bits(&y[name]),"{name} after decode step {step} pool={pool} splits={splits:?}");}
            }
            grouped=next_a;scalar=next_b;token=best_a;
        }
    }
}
#[test]
fn layer_major_prompt_groups_keep_every_logit_and_state_bit() {
    use spark_tp::bounded_decoder::HeadRows;
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let decoder=Decoder::new(&c.admission,Host,8).unwrap();let tokens=[1,7,2,5,3,6,4];
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    for (width,limit) in [(1,8),(2,8),(3,8),(4,8),(5,8),(7,8),(1,2),(2,3),(3,4),(7,1),(4,0)] {
        decoder.set_expanded_limit(limit);
        let mut scalar=decoder.zero_state().unwrap();let mut grouped=decoder.zero_state().unwrap();
        for chunk in tokens.chunks(width) {
            let mut expected=Vec::new();for &token in chunk {let (logits,_,next)=decoder.step(token,&scalar,|_,_|Ok(())).unwrap();scalar=next;expected.push(logits);}
            let (actual,next)=decoder.prefill_rows(chunk,&grouped,HeadRows::All).unwrap();
            for (layer,spark_tp::bounded_decoder::Expanded(value,rows,_)) in &next.expanded {
                let expected=if next.position as usize<=limit {next.position as usize} else {grouped.expanded.get(layer).map_or(0,|v|v.1)};
                assert_eq!(*rows,expected,"retained prefix changed layer={layer} width={width} limit={limit}");
                if let Some(spark_tp::bounded_decoder::Expanded(old,_,_))=grouped.expanded.get(layer) {assert_eq!(bits(old),bits(&value[..old.len()]),"retained prefix bits changed");}
            }
            grouped=next;
            assert_eq!(actual.iter().map(|v|bits(v)).collect::<Vec<_>>(),expected.iter().map(|v|bits(v)).collect::<Vec<_>>(),"width={width}");
            assert_eq!(grouped.position,scalar.position);
            for (a,b) in grouped.layers.iter().zip(&scalar.layers) {assert_eq!(a.keys().collect::<Vec<_>>(),b.keys().collect::<Vec<_>>());for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"{name} width={width}");}}
        }
        // A decode row after a group gives the same bits on either side of the limit.
        let (mut row,row_state)=decoder.prefill_rows(&[9],&grouped,HeadRows::Last).unwrap();let row=row.pop().unwrap();let (step,_,_)=decoder.step(9,&scalar,|_,_|Ok(())).unwrap();assert_eq!(bits(&row),bits(&step),"decode row width={width} limit={limit}");assert_eq!(row_state.position,8);
        decoder.set_expanded_limit(8);
        // Decode continues from the group's state, including its expanded rows.
        let (a,_,next)=decoder.step(9,&grouped,|_,_|Ok(())).unwrap();let (b,_,_)=decoder.step(9,&scalar,|_,_|Ok(())).unwrap();assert_eq!(bits(&a),bits(&b),"decode after width={width}");assert_eq!(next.position,8);
        // The fixed row-count rule gives the same bits on both sides of its limit.
        // A prompt group can follow a decoded token, and only the last head row is optional.
        let (mixed,_,after)=decoder.step(2,&decoder.zero_state().unwrap(),|_,_|Ok(())).unwrap();
        let (last,next)=decoder.prefill_rows(&[1,7],&after,HeadRows::Last).unwrap();assert_eq!(last.len(),1);assert_eq!(next.position,3);
        let (_,_,one)=decoder.step(1,&after,|_,_|Ok(())).unwrap();let (expected,_,_)=decoder.step(7,&one,|_,_|Ok(())).unwrap();
        assert_eq!(bits(&last[0]),bits(&expected));assert_eq!(mixed.len(),expected.len());
    }
}
/// The batch twin: each chat of a batch step gets the logits, the ID and the state bits of its
/// own plain steps, for every set of partner chats, every order and every position mix.
#[test]
#[allow(clippy::needless_range_loop)] // the step index reads every chat's inputs and plain run.
fn batch_steps_keep_each_chats_logits_and_state_bits_with_any_partner() {
    use spark_tp::bounded_decoder::{HeadRows,State};
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let decoder=Decoder::new(&c.admission,Host,8).unwrap();
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let copy=|s:&State<Vec<f32>>|State {position:s.position,layers:s.layers.clone(),expanded:s.expanded.clone()};
    // Three chats at three positions: after a prompt group, after one plain step, and new.
    let prompts:[&[u32];3]=[&[1,7,2],&[5],&[]];
    let starts=prompts.iter().map(|prompt| {
        let mut state=decoder.zero_state().unwrap();
        if prompt.len()>1 {state=decoder.prefill_rows(prompt,&state,HeadRows::Last).unwrap().1;}
        else {for &token in *prompt {state=decoder.step(token,&state,|_,_|Ok(())).unwrap().2;}}
        state
    }).collect::<Vec<_>>();
    let inputs=[[3u32,6,4,2],[4,1,7,3],[2,5,6,1]];
    let mut alone=Vec::new();
    for chat in 0..3 {
        let mut state=copy(&starts[chat]);let mut rows=Vec::new();
        for &token in &inputs[chat] {let (logits,best,next)=decoder.step(token,&state,|_,_|Ok(())).unwrap();rows.push((bits(&logits),best));state=next;}
        alone.push((rows,state));
    }
    for order in [vec![0,1,2],vec![2,1,0],vec![1,2,0],vec![0,1],vec![1,0],vec![0,2],vec![2,1],vec![0],vec![1],vec![2],vec![0,0],vec![1,1,1]] {
        let mut states=order.iter().map(|&chat|copy(&starts[chat])).collect::<Vec<_>>();
        for step in 0..4 {
            let rows=order.iter().zip(&states).map(|(&chat,state)|(inputs[chat][step],state)).collect::<Vec<_>>();
            let results=decoder.step_batch(&rows).unwrap();assert_eq!(results.len(),order.len());
            let mut next=Vec::new();
            for (&chat,(logits,best,state)) in order.iter().zip(results) {
                assert_eq!((bits(&logits),best),alone[chat].0[step],"chat={chat} step={step} order={order:?}");next.push(state);
            }
            states=next;
        }
        for (&chat,state) in order.iter().zip(&states) {
            let expected=&alone[chat].1;assert_eq!(state.position,expected.position);
            for (a,b) in state.layers.iter().zip(&expected.layers) {
                assert_eq!(a.keys().collect::<Vec<_>>(),b.keys().collect::<Vec<_>>());
                for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"{name} chat={chat} order={order:?}");}
            }
        }
    }
    assert!(decoder.step_batch(&[]).is_err());
    let nine=(0..9).map(|_|(1u32,&starts[2])).collect::<Vec<_>>();assert!(decoder.step_batch(&nine).is_err());
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
/// A host compute that keeps expansions by the bits of their latent row, as the
/// device keeps them by origin. It records what decode found and what it added.
#[derive(Default)]
struct Kept {unrecorded:std::cell::RefCell<Vec<bool>>,paused:std::cell::Cell<bool>,released:std::cell::Cell<usize>,keeps:std::cell::Cell<usize>,rows:std::cell::RefCell<std::collections::BTreeMap<(String,Vec<u32>,bool),Vec<f32>>>,found:std::cell::Cell<usize>,largest:std::cell::Cell<usize>,projected:std::cell::Cell<usize>,indirect:std::cell::Cell<usize>,reserved:std::cell::RefCell<Vec<(usize,usize)>>}
impl spark_tp::bounded_decoder::Compute for Kept {
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,s:spark_gpu_probe::causal_ops::Shape,x:[Option<&Self::Value>;4],reserve:usize,expected:f32)->spark_tp::Result<Self::Value> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:spark_gpu_probe::causal_ops::Shape,q:&Self::Value,l:&Self::Value,n:&Self::Value,slots:usize)->spark_tp::Result<Self::Value> {Host.latent_absorption(name,s,q,l,n,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,q:&Self::Value,l:&Self::Value,n:&Self::Value)->spark_tp::Result<Self::Value> {Host.latent_absorption_rows(name,s,q,l,n)}
    fn select_known(&self,ids:&Vec<f32>,count:&Vec<f32>,row:&[f32])->spark_tp::Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->spark_tp::Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,_:usize)->spark_tp::Result<()> {self.prepare(c)}
    type Value=Vec<f32>;type Pass=<Host as spark_tp::bounded_decoder::Compute>::Pass;
    fn rows_project(&self,_:&engine_model::plan::admission::CheckpointAdmission,name:&str,_:&Vec<f32>,m:usize,_:bool)->spark_tp::Result<Option<Vec<f32>>> {
        if name.ends_with("self_attn.kv_b_proj.weight") {self.projected.set(self.projected.get()+m);}Ok(None)
    }
    fn projection_origin(&self,name:&str,x:&Vec<f32>,round:bool)->spark_tp::Result<Option<Vec<f32>>> {
        let found=self.rows.borrow().get(&(name.to_owned(),x.iter().map(|v|v.to_bits()).collect(),round)).cloned();
        if found.is_some() {self.found.set(self.found.get()+1);}Ok(found)
    }
    fn remember_projections(&self,name:&str,round:bool,kept:&[(&Vec<f32>,&Vec<f32>)])->spark_tp::Result<()> {
        self.largest.set(self.largest.get().max(kept.len()));
        for (x,output) in kept {self.rows.borrow_mut().insert((name.to_owned(),x.iter().map(|v|v.to_bits()).collect(),round),(*output).clone());}Ok(())
    }
    fn rows_release(&self)->spark_tp::Result<()> {self.released.set(self.released.get()+1);Ok(())}
    fn rows_keep(&self)->spark_tp::Result<()> {self.keeps.set(self.keeps.get()+1);Ok(())}
    fn rows_unrecorded(&self,pause:bool)->spark_tp::Result<bool> {self.unrecorded.borrow_mut().push(pause);Ok(self.paused.replace(pause))}
    fn upload(&self,v:&[f32])->spark_tp::Result<Vec<f32>> {Host.upload(v)}
    fn read(&self,v:&Vec<f32>)->spark_tp::Result<Vec<f32>> {Host.read(v)}
    fn words(&self,v:&Vec<f32>)->usize {v.len()}
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->spark_tp::Result<Vec<f32>> {Host.range(v,start,len)}
    fn join(&self,v:&[&Vec<f32>])->spark_tp::Result<Vec<f32>> {Host.join(v)}
    fn join_reserved(&self,v:&[&Vec<f32>],reserve:usize)->spark_tp::Result<Vec<f32>> {self.reserved.borrow_mut().push((v.iter().map(|part|part.len()).sum(),reserve));Host.join(v)}
    fn indirect_attention(&self,_:spark_gpu_probe::causal_ops::Shape,_:&[&Vec<f32>],_:&Vec<f32>)->spark_tp::Result<Option<Vec<f32>>> {self.indirect.set(self.indirect.get()+1);Ok(None)}
    fn zeros(&self,n:usize)->spark_tp::Result<Vec<f32>> {Host.zeros(n)}
    fn operation(&self,s:spark_gpu_probe::causal_ops::Shape,x:[Option<&Vec<f32>>;4],w:&[f32])->spark_tp::Result<Vec<f32>> {Host.operation(s,x,w)}
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Vec<f32>,w:&[f32])->spark_tp::Result<Vec<f32>> {Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&Vec<f32>)->spark_tp::Result<(Self::Pass,Vec<f32>)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&Vec<f32>)->spark_tp::Result<Vec<f32>> {Host.finish(p,x)}
    fn mx(&self,p:&spark_tp::mxfp8_projection::Mxfp8Projection<'_>,x:&Vec<f32>)->spark_tp::Result<Vec<f32>> {Host.mx(p,x)}
    fn round(&self,x:&Vec<f32>)->spark_tp::Result<Vec<f32>> {Host.round(x)}
    fn mean(&self,x:&Vec<f32>,s:u32,h:u32)->spark_tp::Result<Vec<f32>> {Host.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Vec<f32>>;5],w:&[f32])->spark_tp::Result<Vec<f32>> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Vec<f32>,m:usize,w:&[f32])->spark_tp::Result<Vec<f32>> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&Vec<f32>,m:usize)->spark_tp::Result<(Self::Pass,Vec<f32>)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&Vec<f32>,m:usize)->spark_tp::Result<Vec<f32>> {Host.rows_finish(p,x,m)}
}
#[test]
fn decode_expansion_mixes_held_kept_and_new_rows_and_keeps_every_bit() {
    use spark_tp::bounded_decoder::HeadRows;
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let plain=Decoder::new(&c.admission,Host,8).unwrap();let tokens=[1,7,2,5,3];let (mut found,mut together)=(0,0);
    // Held rows stop at `limit`; the groups after it add none, and the held ones must stay.
    for (width,limit) in [(2,2),(3,3),(1,1),(5,8),(4,0)] {
        let compute=Kept::default();let kept=Decoder::new(&c.admission,&compute,8).unwrap();kept.set_expanded_limit(limit);
        let mut scalar=plain.zero_state().unwrap();for &token in &tokens {scalar=plain.step(token,&scalar,|_,_|Ok(())).unwrap().2;}
        let mut grouped=kept.zero_state().unwrap();for chunk in tokens.chunks(width) {grouped=kept.prefill_rows(chunk,&grouped,HeadRows::Last).unwrap().1;}
        let held=grouped.expanded.values().map(|rows|rows.1).max().unwrap_or(0);
        assert_eq!(held,if limit>=width {(limit/width*width).min(tokens.len())} else {0},"held rows width={width} limit={limit}");
        // The first decode token expands several rows together; later tokens find them kept.
        for (step,token) in [6,4,9].into_iter().enumerate() {
            let (a,_,next)=kept.step(token,&grouped,|_,_|Ok(())).unwrap();let (b,_,after)=plain.step(token,&scalar,|_,_|Ok(())).unwrap();
            assert_eq!(bits(&a),bits(&b),"decode step {step} width={width} limit={limit}");grouped=next;scalar=after;
        }
        found+=compute.found.get();together=together.max(compute.largest.get());
        assert!(compute.indirect.get()>0,"decode never tried the optional direct consumer");
        assert!(!compute.reserved.borrow().is_empty(),"rejected direct attention never used the reserved join");
        assert!(compute.reserved.borrow().iter().all(|(used,reserve)|used<=reserve && *reserve>0),"fallback join lost its slot reservation");
    }
    // The fixture selects few rows, so the counts are checked over all cases.
    assert!(found>0,"decode never found a kept row");assert!(together>=2,"decode never expanded rows together");
}
#[test]
fn prompt_reuses_held_expansions_without_changing_retention_or_forked_state() {
    use spark_tp::bounded_decoder::HeadRows;
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let tokens=[1,7,2,5,3,6,4];
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let cached=Kept::default();let plain=Kept::default();
    let a=Decoder::new(&c.admission,&cached,8).unwrap();a.set_expanded_limit(2);
    let b=Decoder::new(&c.admission,&plain,8).unwrap();b.set_expanded_limit(0);
    let mut held=a.zero_state().unwrap();let mut scalar=b.zero_state().unwrap();
    for chunk in tokens.chunks(2) {
        let (x,next)=a.prefill_rows(chunk,&held,HeadRows::All).unwrap();let (y,after)=b.prefill_rows(chunk,&scalar,HeadRows::All).unwrap();
        assert_eq!(x.iter().map(|v|bits(v)).collect::<Vec<_>>(),y.iter().map(|v|bits(v)).collect::<Vec<_>>());
        let old=held.expanded.iter().map(|(layer,owner)|(*layer,(owner.1,bits(&owner.0)))).collect::<std::collections::BTreeMap<_,_>>();
        if held.position>0 {let reversed=chunk.iter().rev().copied().collect::<Vec<_>>();let _=a.prefill_rows(&reversed,&held,HeadRows::Last).unwrap();
            assert_eq!(old,held.expanded.iter().map(|(layer,owner)|(*layer,(owner.1,bits(&owner.0)))).collect());}
        held=next;scalar=after;
    }
    // The fork work above is also counted, so use an independent unforked run for cost.
    let counted=Kept::default();let d=Decoder::new(&c.admission,&counted,8).unwrap();d.set_expanded_limit(2);let mut s=d.zero_state().unwrap();
    for chunk in tokens.chunks(2) {s=d.prefill_rows(chunk,&s,HeadRows::Last).unwrap().1;}
    assert!(counted.projected.get()<plain.projected.get(),"retained rows were projected again: {} / {}",counted.projected.get(),plain.projected.get());
    assert!(held.expanded.values().all(|v|v.1==2),"prompt extended the retained count");
}
#[test]
fn verify_passes_keep_only_exact_rows_and_end_in_the_plain_state() {
    use spark_tp::bounded_decoder::Verify;
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let decoder=Decoder::new(&c.admission,Host,8).unwrap();let tokens=[1,7,2,5,3,6,4];
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let mut plain=Vec::new();let mut scalar=decoder.zero_state().unwrap();
    for &token in &tokens {let (logits,_,next)=decoder.step(token,&scalar,|_,_|Ok(())).unwrap();scalar=next;plain.push(bits(&logits));}
    // Each pass: its rows, and how many of them hold the tokens plain decoding chose.
    for passes in [vec![(vec![1,7,2,5,3,6,4],7)],vec![(vec![1,9,9],1),(vec![7,2,9],2),(vec![5,9],1),(vec![3,6,4],3)],vec![(vec![1,7,9],2),(vec![2,9,9],1),(vec![5,9],1),(vec![3,9],1),(vec![6,4],2)],vec![(vec![1],1),(vec![7,2],2),(vec![5,3,9],2),(vec![6],1),(vec![4],1)]] {
        let mut open=Verify::new(decoder.zero_state().unwrap());let mut at=0;
        for (rows,kept) in passes {
            let entry=open.position();assert_eq!(entry,at as u64);let logits=open.rows(&decoder,&rows).unwrap();assert_eq!(logits.len(),rows.len());
            assert_eq!(open.position(),entry+rows.len() as u64);open.truncate(&decoder,entry+kept as u64).unwrap();assert_eq!(open.carried(),0,"kept rows are not computed again");
            for row in &logits[..kept] {assert_eq!(bits(row),plain[at],"row {at}");at+=1;}
            assert!(open.truncate(&decoder,entry+kept as u64+1).is_err(),"a dropped row cannot come back");
        }
        assert_eq!(at,tokens.len());let state=open.settle(&decoder).unwrap();assert_eq!(state.position,scalar.position);
        for (a,b) in state.layers.iter().zip(&scalar.layers) {for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"{name}");}}
    }
    let mut open=Verify::new(decoder.zero_state().unwrap());assert!(open.rows(&decoder,&[]).is_err());assert!(open.truncate(&decoder,1).is_err());
    // A pass of one row as the plain greedy step: the token the logits name, the same state, and
    // nothing done while rows are carried.
    let best=|v:&[u32]|(0..v.len()).reduce(|a,b|if f32::from_bits(v[b])>f32::from_bits(v[a]) {b} else {a}).unwrap() as u32;
    let mut open=Verify::new(decoder.zero_state().unwrap());
    for (at,&token) in tokens.iter().enumerate() {
        if at==3 {
            // Rows 3 and 4 in one pass, the second one wrong: the kept row is not computed again.
            let logits=open.rows(&decoder,&[token,9]).unwrap();assert_eq!(bits(&logits[0]),plain[at]);open.truncate(&decoder,at as u64+1).unwrap();assert_eq!(open.carried(),0);
        } else {assert_eq!(open.row_greedy(&decoder,token).unwrap(),Some(best(&plain[at])),"row {at}");assert_eq!(open.position(),at as u64+1);}
    }
    let state=open.settle(&decoder).unwrap();assert_eq!(state.position,scalar.position);
    for (a,b) in state.layers.iter().zip(&scalar.layers) {for (name,value) in a {assert_eq!(bits(value),bits(&b[name]),"{name}");}}
    // A dropped greedy row leaves no trace: the row then runs again with its logits.
    let mut open=Verify::new(decoder.zero_state().unwrap());assert_eq!(open.row_greedy(&decoder,tokens[0]).unwrap(),Some(best(&plain[0])));
    open.truncate(&decoder,0).unwrap();assert_eq!((open.position(),open.carried()),(0,0));
    assert_eq!(bits(&open.rows(&decoder,&tokens[..1]).unwrap()[0]),plain[0]);
}
#[test]
fn predictor_layer_proposes_without_touching_plain_decoding() {
    let c=fixture::Checkpoint::new_with_predictor(16);
    let plain=Decoder::new(&c.admission,Host,16).unwrap();let mut decoder=Decoder::new(&c.admission,Host,16).unwrap();
    assert!(!decoder.predictor_enabled());decoder.enable_predictor().unwrap();assert!(decoder.predictor_enabled());
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();let vocab=c.admission.description().vocab as u32;
    let (mut a,mut b)=(plain.zero_state().unwrap(),decoder.zero_state().unwrap());let mut seen=None;let mut state_only=None;
    for token in [1u32,7,2,5,3,6] {
        let (expected,_,next)=plain.step(token,&a,|_,_|Ok(())).unwrap();a=next;
        let (logits,chosen,next,hidden)=decoder.step_hidden(token,&b,|_,_|Ok(())).unwrap();b=next;
        assert_eq!(bits(&logits),bits(&expected),"a decoder with the predictor loaded steps the same bits");
        // Proposals are tokens of the vocabulary, the same on every call, with and without the cache.
        for depth in 1..=3 {let first=decoder.propose(&hidden[1],chosen,depth).unwrap();assert_eq!(first.len(),depth);assert!(first.iter().all(|&t|t<vocab));
            assert_eq!(decoder.propose(&hidden[1],chosen,depth).unwrap(),first);}
        let (cached,after)=decoder.propose_after(&hidden[1],chosen,2,seen.as_ref()).unwrap();assert_eq!(cached.len(),2);
        assert_eq!(decoder.propose_after(&hidden[1],chosen,2,seen.as_ref()).unwrap().0,cached,"the cache is not changed by a proposal");
        assert!(decoder.propose_after(&hidden[1],chosen,0,seen.as_ref()).unwrap().0.is_empty());
        // The served form reads no logits: the head's own pick is the scored form's first largest logit, at every depth.
        for depth in 1..=3 {for cache in [None,seen.as_ref()] {let (tokens,scores,_)=decoder.propose_scored(&hidden[1],chosen,depth,cache).unwrap();
            assert_eq!(scores.len(),depth);assert_eq!(decoder.propose_after(&hidden[1],chosen,depth,cache).unwrap().0,tokens,"depth {depth}");}}
        // A row that only brings the cache up (depth 0, the cache half of the layer) leaves the cache a proposal leaves.
        assert_eq!(decoder.propose_after(&hidden[1],chosen,3,state_only.as_ref()).unwrap().0,decoder.propose_after(&hidden[1],chosen,3,seen.as_ref()).unwrap().0);
        state_only=Some(decoder.propose_after(&hidden[1],chosen,0,state_only.as_ref()).unwrap().1);seen=Some(after);
    }
    assert!(decoder.propose(&decoder.zero_state().unwrap().layers[0].values().next().unwrap().clone(),vocab,1).is_err(),"a token outside the vocabulary is refused");
    use spark_tp::bounded_decoder::HeadRows;
    // A pass after steps: with a proposer the held rows follow the decoded rows; the bits stay the plain ones.
    let run=|d:&Decoder<Host>| {let (_,mut state)=d.prefill_rows(&[1,7],&d.zero_state().unwrap(),HeadRows::Last).unwrap();let mut all=Vec::new();
        for token in [2u32,5,3] {let (logits,_,next)=d.step(token,&state,|_,_|Ok(())).unwrap();state=next;all.push(bits(&logits));}
        let (rows,next)=d.prefill_rows(&[6,4],&state,HeadRows::All).unwrap();all.extend(rows.iter().map(|r|bits(r)));
        let (rows,after)=d.prefill_rows(&[2,1],&next,HeadRows::All).unwrap();all.extend(rows.iter().map(|r|bits(r)));
        let (logits,_,_)=d.step(3,&after,|_,_|Ok(())).unwrap();all.push(bits(&logits));
        (all,next.expanded.values().map(|v|v.1).collect::<Vec<_>>(),after.expanded.values().map(|v|v.1).collect::<Vec<_>>())};
    let (expected,held,_)=run(&plain);let (rows,first,second)=run(&decoder);
    assert_eq!(rows,expected,"held decoded rows keep every bit");
    assert!(!held.is_empty() && held.iter().all(|&n|n==2),"without a proposer only the prompt rows are held");
    assert!(first.iter().all(|&n|n==7) && second.iter().all(|&n|n==9),"with a proposer a pass holds every row before its end");
}
#[test]
fn pass_prices_come_only_from_the_requests_own_timed_passes() {
    use spark_tp::bounded_decoder::learned_pass_ms;
    let mut timed=[None;17];
    // Nothing timed: every pass is priced as one step, so the first pass is tried.
    for rows in 1..=16 {assert_eq!(learned_pass_ms(85.0,&timed,rows),85.0);}
    // One count timed: a line from one row to it, on both sides.
    timed[4]=Some(175.0);
    assert_eq!(learned_pass_ms(85.0,&timed,4),175.0);assert_eq!(learned_pass_ms(85.0,&timed,2),115.0);assert_eq!(learned_pass_ms(85.0,&timed,3),145.0);
    assert_eq!(learned_pass_ms(85.0,&timed,6),235.0);assert_eq!(learned_pass_ms(85.0,&timed,40),learned_pass_ms(85.0,&timed,16));
    // Two counts timed: lines between neighbors; a timed count is its own price.
    timed[2]=Some(150.0);
    assert_eq!(learned_pass_ms(85.0,&timed,2),150.0);assert_eq!(learned_pass_ms(85.0,&timed,3),162.5);assert_eq!(learned_pass_ms(85.0,&timed,1),85.0);
    // A pass timed under a step never makes later rows cheaper than it.
    let mut odd=[None;17];odd[3]=Some(60.0);assert_eq!(learned_pass_ms(85.0,&odd,8),60.0);
}
#[test]
fn a_session_first_prompt_group_trims_the_pool_and_later_groups_keep_it() {
    use spark_tp::bounded_decoder::{HeadRows,trims_spares_first,PROMPT_SHARED};
    // The rule: only a group of PROMPT_SHARED (64) rows or more at position 0 trims first.
    assert_eq!(PROMPT_SHARED,64);
    assert!(trims_spares_first(64,0));assert!(trims_spares_first(256,0));
    assert!(!trims_spares_first(63,0),"small groups keep their existing trim");
    assert!(!trims_spares_first(256,256),"later groups keep the previous group's spares");
    // Through the decoder: a trim (keep) before the first 64-row group and after it, no release.
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,160);let _=&c.operands;
    let compute=Kept::default();let decoder=Decoder::new(&c.admission,&compute,160).unwrap();
    let tokens=(0..128u32).map(|t|t%7+1).collect::<Vec<_>>();
    let first=decoder.prefill_rows(&tokens[..64],&decoder.zero_state().unwrap(),HeadRows::Last).unwrap().1;
    assert_eq!((compute.released.get(),compute.keeps.get()),(0,2));
    let _=decoder.prefill_rows(&tokens[64..],&first,HeadRows::Last).unwrap();
    assert_eq!((compute.released.get(),compute.keeps.get()),(0,3),"a later group keeps once, at its end");
    let _=decoder.prefill_rows(&tokens[..2],&decoder.zero_state().unwrap(),HeadRows::Last).unwrap();
    assert_eq!((compute.released.get(),compute.keeps.get()),(0,3),"a small first group neither trims nor keeps");
}
/// The absorbed decode path takes the selection count from the context and checks the device's word later.
/// The formula must equal the known-selection row wherever that row exists, and keep the full budget above it.
#[test]
fn selected_count_equals_the_known_selection_count_and_holds_above_the_budget() {
    use spark_gpu_probe::{prompt_ops::{Op,Shape},select_admission::known_selection};
    use spark_tp::bounded_decoder::selected_count;
    for (pool,budget,tail) in [(4u64,2048u64,true),(4,2048,false),(1,3,true),(8,16,true),(2,4,false)] {
        let slots=budget+if tail {pool-1} else {0};
        for context in 1..=(budget+3*pool+5) {
            let mut s=Shape::new(Op::SelectRows,1).unwrap();s.dim=pool as u32;s.taps=budget as u32;s.cols=u32::from(tail);s.slots=slots as u32;s.first=context as u32;s.rows=(context/pool) as u32;
            let count=selected_count(pool,budget,tail,context);assert!(count<=slots.min(context),"pool={pool} budget={budget} tail={tail} context={context}");
            if let Some(row)=known_selection(s) {assert_eq!(row[slots as usize],count as f32,"pool={pool} budget={budget} tail={tail} context={context}");}
            else {assert_eq!(count,budget+if tail {context%pool} else {0},"pool={pool} budget={budget} tail={tail} context={context}");}
        }
    }
    assert_eq!(selected_count(4,2048,true,4158),2050);assert_eq!(selected_count(4,2048,true,15),15);assert_eq!(selected_count(0,2048,true,15),0);
}
#[test]
fn every_prompt_group_runs_unrecorded_and_restores_recording() {
    use spark_tp::bounded_decoder::HeadRows;
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let plain=Decoder::new(&c.admission,Host,8).unwrap();let compute=Kept::default();let d=Decoder::new(&c.admission,&compute,8).unwrap();
    let (mut a,mut b)=(plain.zero_state().unwrap(),d.zero_state().unwrap());
    for chunk in [1u32,7,2,5,3].chunks(2) {
        let (la,na)=plain.prefill_rows(chunk,&a,HeadRows::Last).unwrap();let (lb,nb)=d.prefill_rows(chunk,&b,HeadRows::Last).unwrap();
        assert_eq!(la.iter().map(|l|bits(l)).collect::<Vec<_>>(),lb.iter().map(|l|bits(l)).collect::<Vec<_>>());a=na;b=nb;
    }
    assert_eq!(*compute.unrecorded.borrow(),[true,false].repeat(3),"each group pauses recording once and restores it");assert!(!compute.paused.get());
    // A caller that already paused keeps its pause after the group.
    compute.paused.set(true);d.prefill_rows(&[4],&b,HeadRows::Last).unwrap();assert!(compute.paused.get());
    // A failed group restores recording too.
    compute.paused.set(false);assert!(d.prefill_rows(&[u32::MAX],&b,HeadRows::Last).is_err());assert!(!compute.paused.get());
}
