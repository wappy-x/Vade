//! Two chats use the same grouped expert path as larger cohorts. Every output and state stays
//! equal to its own plain step, including copies and chats that leave and rejoin at new positions.
use spark_tp::bounded_decoder::{Decoder,Host,expert_union_counts};
use engine_model::{safetensors::Dtype,shapes::Ffn};
#[allow(dead_code,reason="shared checkpoint supports more forms than this test needs")]
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn grouped_pairs_keep_each_chats_bits_and_reuse_copied_experts() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,64);
    let d=Decoder::new(&c.admission,Host,64).unwrap();
    let mut states=(0..8).map(|_|d.zero_state().unwrap()).collect::<Vec<_>>();
    let routed=c.admission.description().layers.iter().enumerate().filter_map(|(i,l)|match l.ffn {Ffn::Experts(e)=>Some((i,e.selected)),_=>None}).collect::<Vec<_>>();
    assert!(!routed.is_empty());
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    for (turn,width) in (2..=8).chain((1..8).rev()).chain(2..=8).enumerate() {
        let slots=(0..width).map(|i|(i+turn)%8).collect::<Vec<_>>();
        let tokens=slots.iter().map(|&i|1+((i+turn)%7) as u32).collect::<Vec<_>>();
        let rows=slots.iter().zip(&tokens).map(|(&i,&token)|(token,&states[i])).collect::<Vec<_>>();
        let before=expert_union_counts();let output=d.step_batch(&rows).unwrap();let after=expert_union_counts();
        for &(layer,k) in &routed {
            assert_eq!(after[layer][0]-before[layer][0],if width>=2 {width as u64*k} else {0},"grouped picks width={width}");
        }
        for ((&slot,&token),(logits,id,next)) in slots.iter().zip(&tokens).zip(output) {
            let (plain,expected,want)=d.step(token,&states[slot],|_,_|Ok(())).unwrap();
            assert_eq!((bits(&logits),id),(bits(&plain),expected),"width={width} slot={slot}");
            assert_eq!(next.position,want.position);
            for (got,want) in next.layers.iter().zip(&want.layers) {
                assert_eq!(got.keys().collect::<Vec<_>>(),want.keys().collect::<Vec<_>>());
                for (name,v) in got {assert_eq!(bits(v),bits(&want[name]),"width={width} slot={slot} {name}");}
            }
            states[slot]=next;
        }
    }
    let before=expert_union_counts();let output=d.step_batch(&[(3,&states[0]),(3,&states[0])]).unwrap();let after=expert_union_counts();
    for &(layer,k) in &routed {assert_eq!([after[layer][0]-before[layer][0],after[layer][1]-before[layer][1]],[2*k,k],"copied expert union");}
    assert_eq!(bits(&output[0].0),bits(&output[1].0));assert_eq!(output[0].1,output[1].1);
    // Equal copies could be wrong together: each must also be the chat alone, logits, ID and every state plane.
    let (plain,expected,want)=d.step(3,&states[0],|_,_|Ok(())).unwrap();
    for (logits,id,next) in &output {
        assert_eq!((bits(logits),*id,next.position),(bits(&plain),expected,want.position),"a copy differs from the chat alone");
        for (got,want) in next.layers.iter().zip(&want.layers) {for (name,v) in got {assert_eq!(bits(v),bits(&want[name]),"copy state {name}");}}
    }
}
