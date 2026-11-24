//! Host tests of the expert half, on a made-up checkpoint. The expected words come from a scalar form in this file.
use super::*;
use super::super::mix::support::{*,hand};
use engine_model::plan::description::LoopTextDescription;
use spark_gpu_probe::fixed_math::{rn,sigmoid};
use std::collections::VecDeque;
mod on_fixture;
mod host_form;
mod real_weights;

/// The scalar form of step 10: the picked expert numbers in pick order, their weights, and the summed expert output.
fn expert_sum(d:&LoopTextDescription,w:&Made,layer:usize,input:&[f32])->(Vec<usize>,Vec<f32>,Vec<f32>) {
    let Ffn::Experts(experts)=d.layers[layer].ffn else {panic!("routed experts")};let inner=experts.width as usize;
    let z=hand::rounded_norm(input,w.get(Weight::ExpertNorm(layer)),d.epsilon as f32);
    let scores=hand::product(w.get(Weight::Router(layer)),&z).into_iter().map(sigmoid).collect::<Vec<_>>();
    let mut choice=scores.iter().zip(w.get(Weight::RouterBias(layer))).map(|(score,bias)|score+bias).collect::<Vec<_>>();
    let (mut ids,mut weights,mut total)=(Vec::new(),Vec::new(),0.0f32);
    for _ in 0..experts.selected {
        // A strict comparison in rising expert number keeps the lower number on equal values.
        let (mut best,mut id)=(f32::NEG_INFINITY,0);for (expert,&value) in choice.iter().enumerate() {if value>best {best=value;id=expert;}}
        choice[id]=f32::NEG_INFINITY;total+=scores[id];ids.push(id);weights.push(scores[id]);
    }
    for weight in &mut weights {*weight=*weight/(total+1e-20)*d.routing.scale as f32;}
    let downs=ids.iter().map(|&expert|{
        let both=hand::product(w.get(Weight::GateUp(layer,expert)),&z);
        let active=(0..inner).map(|i|hand::active(both[i],both[inner+i],(hand::inverse(d.caps.gate),d.caps.gate as f32),(hand::inverse(d.caps.up),d.caps.up as f32))).collect::<Vec<_>>();
        hand::product(w.get(Weight::Down(layer,expert)),&active)
    }).collect::<Vec<_>>();
    let out=hand::fold(&weights,&downs.iter().map(Vec::as_slice).collect::<Vec<_>>());
    (ids,weights,out)
}
fn visit_of(d:&LoopTextDescription,layer:u64)->Visit {*d.visits.iter().find(|v|v.layer==layer).unwrap()}

#[test]
fn the_expert_sum_equals_the_scalar_form() {
    let d=small();let width=d.hidden as usize;let mut n=Numbers::new(21);let mut all=std::collections::BTreeSet::new();
    for layer in 0..d.layers.len() {for round in 0..4 {
        let w=Made::new(&d,22+round);let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();
        let input=n.words(width,3.0);
        let (ids,_,want)=expert_sum(&d,&w,layer,&input);
        let got=sum(&run,&visit_of(&d,layer as u64),&input).unwrap();
        assert_eq!(bits(&got),bits(&want),"layer {layer} round {round}");
        assert!(got.iter().all(|&v|rn(v).to_bits()==v.to_bits()) && got.iter().any(|&v|v!=0.0),"the sum holds BF16 values");
        // The products run in this order: the router, each picked gate-and-up matrix, each picked down matrix.
        let mut order=vec![Weight::Router(layer)];order.extend(ids.iter().map(|&e|Weight::GateUp(layer,e)));order.extend(ids.iter().map(|&e|Weight::Down(layer,e)));
        assert_eq!(*w.asked.borrow(),order);
        // One read a visit: the picked expert numbers. One rounded norm, one route, one activation, one fold.
        assert_eq!((c.reads.get(),c.count(Op::RoundedNorm),c.count(Op::Route),c.count(Op::CappedGate),c.count(Op::RouteFold)),(1,1,1,1,1));assert_eq!(c.norms.get(),0);assert!(c.standing.borrow().is_empty());
        assert_eq!(ids.len(),3);all.extend(ids);
    }}
    // The made-up numbers reach every expert, so the pick is not one fixed set.
    assert_eq!(all.len(),5);
}
#[test]
fn equal_scores_pick_the_lower_expert_first_and_the_bias_only_picks() {
    let d=small();let width=d.hidden as usize;let input=Numbers::new(23).words(width,3.0);
    let mut w=Made::new(&d,24);w.words.get_mut(&Weight::Router(0)).unwrap().fill(0.0);w.words.get_mut(&Weight::RouterBias(0)).unwrap().fill(0.0);
    let run_one=|w:&Made|->(Vec<Weight>,Vec<u32>) {let c=Stand::new();let run=Run::new(&c,w,&d).unwrap();
        let out=bits(&sum(&run,&d.visits[0],&input).unwrap());(w.asked.borrow()[1..4].to_vec(),out)};
    // Every score is 0.5 and every bias is 0: the three lowest numbers, in rising order.
    let (asked,even)=run_one(&w);assert_eq!(asked,[Weight::GateUp(0,0),Weight::GateUp(0,1),Weight::GateUp(0,2)]);
    let (ids,weights,want)=expert_sum(&d,&w,0,&input);assert_eq!((ids,bits(&want)),(vec![0,1,2],even));
    // Each weight is 0.5 / 1.5 * the route scale: the three add up to the scale.
    assert!(weights.iter().all(|&v|(v-2.0/3.0).abs()<1e-6));
    // A bias moves expert 4 to the front. The scores, and so the weights, do not change.
    w.asked.borrow_mut().clear();w.words.get_mut(&Weight::RouterBias(0)).unwrap()[4]=1.0;
    let (asked,_)=run_one(&w);assert_eq!(asked,[Weight::GateUp(0,4),Weight::GateUp(0,0),Weight::GateUp(0,1)]);
    let (ids,biased,_)=expert_sum(&d,&w,0,&input);assert_eq!(ids,[4,0,1]);assert_eq!(bits(&biased),bits(&weights));
}
#[test]
fn the_route_words_are_checked_before_they_choose_a_weight() {
    assert_eq!(picked(&[4.0,0.0,2.0],5).unwrap(),[4,0,2]);
    for bad in [[5.0,0.0,1.0],[1.5,0.0,2.0],[-1.0,0.0,2.0],[2.0,0.0,2.0],[f32::NAN,0.0,1.0],[f32::INFINITY,0.0,1.0]] {assert!(picked(&bad,5).is_err(),"{bad:?}");}
}
#[test]
fn the_convolution_scale_is_the_64_bit_reciprocal_rounded_once() {
    assert_eq!(conv_scale(5).to_bits(),((1.0f64/5.0f64.sqrt()) as f32).to_bits());
    assert_eq!(conv_scale(9).to_bits(),((1.0f64/3.0) as f32).to_bits());assert_eq!(conv_scale(4),0.5);
    // It is a product with the reciprocal, not a division: the two differ on some input.
    let scale=conv_scale(5);let root=5.0f32.sqrt();
    assert!((1..2000).any(|i|{let x=i as f32*0.37;(x*scale).to_bits()!=(x/root).to_bits()}));
}
/// A chat of 12 tokens through the expert half of one visit. The ring has one row a tap and the slot of position `p`
/// is `p mod taps`, so the chat crosses the ring edge twice. The scalar form keeps the last inputs in a queue.
#[test]
fn the_expert_half_runs_a_chat_across_the_ring_edge() {
    let d=small();let width=d.hidden as usize;let taps=d.conv.kernel as usize;assert_eq!(taps,4);
    for layer in [0u64,3] {
        let visit=visit_of(&d,layer);let layer=layer as usize;
        let w=Made::new(&d,25);let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();let mut n=Numbers::new(26);
        let mut ring=vec![0.0f32;taps*width];let mut history=VecDeque::from(vec![vec![0.0f32;width];taps]);
        let by_lag=w.get(Weight::ConvAfterExperts(layer));
        for position in 0..12usize {
            let (stream,residual)=(n.bf16(width,2.0),n.words(width,2.0));let slot=position%taps;
            // The driver's half. Layer 0 starts a block; layer 3 does not. Each gets a new token state, as a token step does.
            let mut token=Token::new();let mut list:Vec<Vec<f32>>=Vec::new();
            if !d.layers[layer].boundary {
                // A layer inside a block reads a list that an earlier layer filled: here one made-up block vector.
                let first=n.bf16(width,2.0);let start=d.visits[0];
                token.expert_input(&run,&start,&first,&first).unwrap();list.push(first);
            } else {list.push(stream.clone());}
            let got=half(&run,&visit,&mut token,&stream,&residual,|u|{
                ring[slot*width..(slot+1)*width].copy_from_slice(u);Ok((ring.clone(),vec![slot as f32]))}).unwrap();
            // The scalar form.
            let entries=list.iter().map(Vec::as_slice).chain([&residual[..]]).collect::<Vec<_>>();
            let (_,_,u)=expert_sum(&d,&w,layer,&hand::mix(w.get(Weight::MixExperts(layer)),&entries));
            history.pop_front();history.push_back(u);
            let want=(0..width).map(|channel|{
                let past=history.iter().map(|row|row[channel]).collect::<Vec<_>>();
                rn(residual[channel]+hand::conv(&past,&by_lag[channel*taps..(channel+1)*taps],conv_scale(taps as u64+1)))
            }).collect::<Vec<_>>();
            assert_eq!(bits(&got),bits(&want),"layer {layer} position {position}");
            assert!(got.iter().all(|&v|rn(v).to_bits()==v.to_bits()),"the stream after a visit holds BF16 values");
        }
        assert_eq!(c.count(Op::ShortConv),12);assert!(c.standing.borrow().is_empty() && c.compared.get()>=12);
    }
}
#[test]
fn the_taps_meet_the_inputs_by_their_lag() {
    // One channel that matters: taps 1, 10, 100, 1000 by lag, and the inputs 1, 2, 3, 4 with 4 the current token.
    // Sum, oldest first: 1000 * 1 + 100 * 2 + 10 * 3 + 1 * 4, then the current input once more: 1238.
    let d=small();let width=d.hidden as usize;let taps=4;let mut w=Made::new(&d,27);
    let weight=w.words.get_mut(&Weight::ConvAfterExperts(0)).unwrap();weight.fill(0.0);weight[..taps].copy_from_slice(&[1.0,10.0,100.0,1000.0]);
    let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();
    // Position 5: slot 1 holds the current input, slot 0 the one before it, slot 3 the one before that, slot 2 the oldest.
    let mut ring=vec![0.0f32;taps*width];for (slot,value) in [(2,1.0),(3,2.0),(0,3.0),(1,4.0)] {ring[slot*width]=value;}
    let residual=vec![0.25f32;width];
    let out=close(&run,&d.visits[0],&ring,&vec![1.0],0,&residual).unwrap();
    assert_eq!(out[0],rn(0.25+rn(1238.0*conv_scale(5))));
    // A channel with no tap keeps its own input: (0 + u) * scale. Here u is 0, so the residual stays.
    assert!(out[1..].iter().all(|&v|v==0.25));
    // The convolution is rounded before the sum, and the sum is rounded once more.
    assert_ne!(rn(0.25+1238.0*conv_scale(5)).to_bits(),0);assert_eq!(rn(out[0]).to_bits(),out[0].to_bits());
}
#[test]
fn a_wrong_extent_is_refused() {
    let d=small();let w=Made::new(&d,28);let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();let width=d.hidden as usize;
    let (good,short)=(vec![1.0f32;width],vec![1.0f32;width-1]);let ring=vec![0.0f32;4*width];let visit=d.visits[0];
    assert!(sum(&run,&visit,&short).unwrap_err().contains("no whole count of rows of the width"));
    assert!(close(&run,&visit,&ring[width..].to_vec(),&vec![0.0],0,&good).is_err());
    assert!(close(&run,&visit,&ring,&vec![0.0,0.0],0,&good).is_err());
    assert!(close(&run,&visit,&ring,&vec![0.0],0,&short).is_err());
    assert!(close(&run,&visit,&ring,&vec![0.0],0,&good).is_ok());
    // A slot outside the ring stops the launch.
    assert!(close(&run,&visit,&ring,&vec![4.0],0,&good).is_err());
    // A layer with no routed experts, a router bias of another length, and a visit outside the layers.
    let mut dense=d.clone();dense.layers[0].ffn=Ffn::Dense {width:8};
    assert!(sum(&Run::new(&c,&w,&dense).unwrap(),&visit,&good).unwrap_err().contains("routed experts"));
    let mut cut=Made::new(&d,28);cut.words.get_mut(&Weight::RouterBias(0)).unwrap().pop();
    assert!(sum(&Run::new(&c,&cut,&d).unwrap(),&visit,&good).unwrap_err().contains("router bias"));
    let mut outside=visit;outside.layer=99;assert!(sum(&run,&outside,&good).is_err());
}
