//! Shared plain steps retain each chat's own predictor inputs and cache.
use spark_tp::bounded_decoder::{Decoder,Host,Verify,PROPOSER_HIDDEN};
#[allow(dead_code,reason="shared checkpoint fixture")]
#[path="support/attention_checkpoint.rs"] mod fixture;
#[test]
fn tracked_batch_rows_keep_hidden_bits_and_probe_history_of_each_chat_alone() {
    let c=fixture::Checkpoint::new_with_predictor(48);
    let mut d=Decoder::new(&c.admission,Host,48).unwrap();d.enable_predictor().unwrap();
    let mut chats=(0..2).map(|_|{let mut v=Verify::new(d.zero_state().unwrap());v.track_proposer();v}).collect::<Vec<_>>();
    let mut twins=(0..2).map(|_|{let mut v=Verify::new(d.zero_state().unwrap());v.track_proposer();v}).collect::<Vec<_>>();
    let mut states=(0..2).map(|_|d.zero_state().unwrap()).collect::<Vec<_>>();
    for step in 0..20 {
        let tokens=[(step%9+1) as u32,((step*3+4)%11+1) as u32];
        let rows=chats.iter_mut().zip(tokens).map(|(v,t)|(t,v.batch_state().unwrap())).collect::<Vec<_>>();
        let (results,hidden)=d.step_batch_tracked(&rows,false).unwrap();
        for (at,((logits,best,next),hidden)) in results.into_iter().zip(hidden).enumerate() {
            let (expected,id,state,own_hidden)=d.step_hidden(tokens[at],&states[at],|_,_|Ok(())).unwrap();states[at]=state;
            assert_eq!(best,id);assert_eq!(logits.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
            assert_eq!(hidden.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),own_hidden[PROPOSER_HIDDEN].iter().map(|x|x.to_bits()).collect::<Vec<_>>());
            chats[at].batch_advanced(tokens[at],next,hidden).unwrap();twins[at].rows(&d,&[tokens[at]]).unwrap();
            if step==7 || step==15 {assert_eq!(chats[at].probe(&d,best,3).unwrap().0,twins[at].probe(&d,best,3).unwrap().0);}
            assert_eq!(chats[at].waiting(),twins[at].waiting());
        }
    }
    for at in 0..2 {assert_eq!(chats[at].propose(&d,5,3).unwrap().0,twins[at].propose(&d,5,3).unwrap().0);}
}
