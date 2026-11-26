//! The expert half on the two made-up checkpoints of the fixture file: a chat of 12 tokens through every visit.
//!
//! The driver's mix calls give the input of each expert half (the mix tests compare those words with a scalar
//! replay on the same fixtures). Here each expert half is compared with the scalar form of this folder, visit by
//! visit, and so is the state that a visit keeps for the next token: the ring of the convolution after the experts.
//! A made-up vector stands in for the output of the attention half. The ring write is the rule of the state file
//! (the slot of position `p` is `p mod taps`); this test writes a host ring by that rule.
use super::*;
use super::super::super::fixture::Fixture;
use std::collections::BTreeSet;

fn chat(fixture:&Fixture,seed:u64) {
    let d=&fixture.description;let (width,taps,visits)=(d.hidden as usize,d.conv.kernel as usize,d.visits.len());
    let w=Made::stored(fixture);let c=Stand::new();let run=Run::new(&c,&w,d).unwrap();let mut n=Numbers::new(seed);
    let table=&fixture.stored("model.llm.embed.weight").unwrap().words;
    // The state of each visit: the ring that the driver reads, and the last inputs of the scalar form, oldest first.
    let mut rings=vec![vec![0.0f32;taps*width];visits];
    let mut histories=vec![VecDeque::from(vec![vec![0.0f32;width];taps]);visits];
    let mut picks=BTreeSet::new();
    assert!(taps<12,"the chat must cross the ring edge");
    for (position,&id) in fixture.chat(12).iter().enumerate() {
        let slot=position%taps;let mut token=Token::new();
        let mut stream=table[id as usize*width..(id as usize+1)*width].to_vec();
        for (index,visit) in d.visits.iter().enumerate() {
            let layer=visit.layer as usize;let at=format!("position {position} visit {index} layer {layer}");
            let keep=token.enter(&run,visit,stream).unwrap();token.attention_inputs(&run,visit,&keep).unwrap();
            let attended=n.bf16(width,1.5);
            let residual=if d.layers[layer].boundary {attended} else {keep.iter().zip(&attended).map(|(a,b)|a+b).collect()};
            let input=token.expert_input(&run,visit,&keep,&residual).unwrap();
            // The driver: the expert sum, the ring write, the convolution and the stream after the visit.
            let u=sum(&run,visit,&input).unwrap();
            rings[index][slot*width..(slot+1)*width].copy_from_slice(&u);
            let got=close(&run,visit,&rings[index],&vec![slot as f32],0,&residual).unwrap();
            // The scalar form.
            let (ids,_,want_u)=expert_sum(d,&w,layer,&input);picks.extend(ids.into_iter().map(|expert|(layer,expert)));
            assert_eq!(bits(&u),bits(&want_u),"expert sum, {at}");
            histories[index].pop_front();histories[index].push_back(want_u);
            let by_lag=w.get(Weight::ConvAfterExperts(layer));
            let want=(0..width).map(|channel|{
                let past=histories[index].iter().map(|row|row[channel]).collect::<Vec<_>>();
                rn(residual[channel]+hand::conv(&past,&by_lag[channel*taps..(channel+1)*taps],conv_scale(taps as u64+1)))
            }).collect::<Vec<_>>();
            assert_eq!(bits(&got),bits(&want),"stream after the visit, {at}");
            assert!(got.iter().all(|&v|rn(v).to_bits()==v.to_bits()),"the stream after a visit holds BF16 values, {at}");
            stream=got;
        }
        assert_eq!(token.end_mix(&run,&stream).unwrap().len(),width);
        // The state bytes of each visit after this token: the row of slot `q mod taps` holds the expert sum of
        // position `q` for the last `taps` positions, and a slot that no token wrote is zero.
        for (index,ring) in rings.iter().enumerate() {
            for (age,row) in histories[index].iter().rev().enumerate() {
                let at=(slot+taps-age)%taps;
                assert_eq!(bits(&ring[at*width..(at+1)*width]),bits(row),"ring of visit {index} after position {position}, age {age}");
            }
        }
    }
    // One convolution, one route (the one-row `RouterPick`) and one fold a visit and a token. Each ran on its arm
    // with the bits of its hand form.
    assert_eq!((c.count(Op::ShortConv),c.count(Op::RouterPick),c.count(Op::RouteFold)),(12*visits,12*visits,12*visits));
    assert!(c.standing.borrow().is_empty() && c.norms.get()==0);
    // The pick follows the data: over the chat a layer picks more experts than one token can.
    let Ffn::Experts(experts)=d.layers[0].ffn else {panic!("routed experts")};
    assert!((0..d.layers.len()).any(|layer|picks.iter().filter(|pick|pick.0==layer).count()>experts.selected as usize));
}
#[test]
fn a_chat_of_12_tokens_runs_through_every_visit_of_the_small_fixture() {chat(&Fixture::small(),41);}
#[test]
fn a_chat_of_12_tokens_runs_through_every_visit_of_the_wide_fixture() {chat(&Fixture::wide(),42);}
