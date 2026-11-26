//! The mix and the expert half against the whole-model host form of the model crate (`engine_model::loop_host`),
//! word for word, on the two made-up checkpoints of the fixture file and a chat of 12 tokens.
//!
//! The host form shares no code with the driver: it reads the stored tensors itself and runs the host form of each
//! operation. It gives the values inside each visit. This test takes from it only what the attention half makes:
//! the stream after attention of each visit (the residual). Every other value is made by the driver and compared:
//! - the three attention inputs after their norms (steps 2 to 4 of a visit: the mix, the messages and the gates);
//! - the sum of the picked experts (steps 7 to 10) and the experts that the router picked;
//! - the stream after the visit (steps 11 and 12), which is the input of the next visit;
//! - the end mix of the token;
//! - the state that each visit keeps: the last inputs of the convolution after the experts.
use super::*;
use super::super::super::{fixture::Fixture,mix::Inputs};
use engine_model::{loop_host::{Head,Limits,Model,Want},short_conv::Slot};

fn compare(fixture:&Fixture)->usize {
    let d=&fixture.description;let (width,taps,visits)=(d.hidden as usize,d.conv.kernel as usize,d.visits.len());
    let model=Model::new(d,&fixture.index,Box::new(fixture.memory()),Limits {expert_cache_bytes:usize::MAX}).unwrap();
    let runner=model.runner().unwrap();let mut chat=runner.chat().unwrap();let tokens=fixture.chat(12);
    let outs=runner.read(&mut chat,&tokens,Want {head:Head::None,records:true},1).unwrap();
    let w=Made::stored(fixture);let c=Stand::new();let run=Run::new(&c,&w,d).unwrap();
    let norm_weight=|weight:Weight|<Fixture as Weights<Stand>>::words(fixture,weight).unwrap();
    let mut rings=vec![vec![0.0f32;taps*width];visits];let mut words=0;
    for (position,out) in outs.iter().enumerate() {
        assert_eq!((out.streams.len(),out.records.len()),(visits,visits));
        let slot=position%taps;let mut token=Token::new();let mut stream=out.embedding.clone();
        for (index,visit) in d.visits.iter().enumerate() {
            let (record,layer)=(&out.records[index],visit.layer as usize);let at=format!("position {position} visit {index} layer {layer}");
            let keep=token.enter(&run,visit,stream).unwrap();
            let inputs=match token.attention_inputs(&run,visit,&keep).unwrap() {Inputs::Same(one)|Inputs::NormedSame(one)|Inputs::NormedHalfSame(one)=>[one.clone(),one.clone(),one],Inputs::Each(each)|Inputs::NormedEach(each)|Inputs::NormedHalfEach(each)=>each};
            for (which,(input,weight)) in inputs.iter().zip([Weight::QueryNorm(layer),Weight::KeyNorm(layer),Weight::ValueNorm(layer)]).enumerate() {
                assert_eq!(bits(&run.rounded_norm(input,norm_weight(weight)).unwrap()),bits(&record.attention_inputs[which]),"attention input {which}, {at}");
            }
            w.asked.borrow_mut().clear();
            let input=token.expert_input(&run,visit,&keep,&record.residual).unwrap();
            let u=sum(&run,visit,&input).unwrap();
            let picked=w.asked.borrow().iter().filter_map(|asked|match asked {Weight::GateUp(_,expert)=>Some(*expert),_=>None}).collect::<Vec<_>>();
            assert_eq!(picked,record.route.ids,"picked experts, {at}");
            assert_eq!(bits(&u),bits(&record.experts),"expert sum, {at}");
            rings[index][slot*width..(slot+1)*width].copy_from_slice(&u);
            let got=close(&run,visit,&rings[index],&vec![slot as f32],0,&record.residual).unwrap();
            assert_eq!(bits(&got),bits(&out.streams[index]),"stream after the visit, {at}");
            words+=3*width+2*width;stream=got;
        }
        assert_eq!(bits(&token.end_mix(&run,&stream).unwrap()),bits(&out.end),"end mix, position {position}");words+=width;
    }
    // The state of each visit after the chat: the host form keeps the last `taps - 1` inputs, the oldest first.
    for (index,state) in chat.visits().iter().enumerate() {
        let kept=state.conv.get(Slot::AfterExperts).inputs();assert_eq!(kept.len(),(taps-1)*width);
        for (row,want) in kept.chunks_exact(width).enumerate() {
            let position=tokens.len()-(taps-1)+row;let slot=position%taps;
            assert_eq!(bits(&rings[index][slot*width..(slot+1)*width]),bits(want),"ring of visit {index}, position {position}");
        }
    }
    assert!(c.standing.borrow().is_empty() && c.norms.get()==0);
    words
}
#[test]
fn the_two_halves_equal_the_whole_model_host_form_on_the_small_fixture() {
    let fixture=Fixture::small();let d=&fixture.description;
    assert_eq!(compare(&fixture),12*(d.visits.len()*5+1)*d.hidden as usize);
}
#[test]
fn the_two_halves_equal_the_whole_model_host_form_on_the_wide_fixture() {
    let fixture=Fixture::wide();let d=&fixture.description;
    assert_eq!(compare(&fixture),12*(d.visits.len()*5+1)*d.hidden as usize);
}
