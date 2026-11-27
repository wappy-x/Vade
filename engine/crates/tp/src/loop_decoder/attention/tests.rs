//! Tests of the attention half: the numbers it makes from the config, each arm against its hand form, and one visit
//! on the two made-up checkpoints. The whole chat against the whole-model host form is in the tests of the driver.
use super::{*,hand::{self,Hand,Mode,bits,unpacked}};
use crate::{bounded_decoder::Host,loop_decoder::{fixture::{Fixture,made},mix::{Inputs,Run,Weight,Weights},state::{self,Ring}}};
use spark_gpu_probe::{causal_ops::{Op,Shape},fixed_math::{ln,rn}};
use std::collections::BTreeMap;

fn shape(op:Op,heads:usize,dim:usize,taps:usize,rows:usize,cols:usize,epsilon:f32,bound:f32)->Shape {
    Shape {op,heads:heads as u32,dim:dim as u32,taps:taps as u32,rows:rows as u32,cols:cols as u32,epsilon,bound}
}
/// Rows of BF16 values, packed two a word.
fn packed(values:&[f32])->Vec<f32> {values.as_chunks::<2>().0.iter().map(|pair|f32::from_bits((pair[0].to_bits()>>16)|(pair[1].to_bits()&0xffff_0000))).collect()}
#[test]
fn the_numbers_from_the_config_are_made_in_64_bit_and_rounded_once() {
    // 128^(-1/2) = 0.08838834764831845: the 32-bit value, not a BF16 value.
    assert_eq!(score_scale(128).to_bits(),0x3db5_04f3);assert_ne!(rn(score_scale(128)).to_bits(),score_scale(128).to_bits());
    assert_eq!((score_scale(4),score_scale(36).to_bits()),(0.5,((1.0f64/6.0) as f32).to_bits()));
    // The factor is 1 up to the floor: the first five tokens of floor 5.
    let scale=PositionScale {floor:5,alpha:0.25};
    for position in 0..5 {assert_eq!(position_factor(scale,position).to_bits(),1.0f32.to_bits(),"position {position}");}
    let fifth=(1.0f64/5.0) as f32;
    for position in [5u64,9,11,1000] {assert_eq!(position_factor(scale,position).to_bits(),(1.0+0.25f32*ln((position+1) as f32*fifth)).to_bits());}
    assert!(position_factor(scale,9)>1.17 && position_factor(scale,9)<1.18,"1 + 0.25 * ln(2)");
    // A product with the reciprocal is not a division: some position tells the two apart, and the factor follows the product.
    let (third,scale)=((1.0f64/3.0) as f32,PositionScale {floor:3,alpha:0.5});
    let apart=(0..4096u64).filter(|&p|((p+1) as f32/3.0).to_bits()!=((p+1) as f32*third).to_bits()).collect::<Vec<_>>();
    assert!(!apart.is_empty());
    for &position in &apart {assert_eq!(position_factor(scale,position).to_bits(),(1.0+0.5f32*ln(((position+1) as f32*third).max(1.0))).to_bits());}
    assert!(position_factor(scale,u64::MAX).is_finite());
}
/// Launches of every arm that the attention half and the token step run, with made-up words. The sizes cross the
/// 32 lanes of a sum, the 128 values of a norm chunk and the 1,024 distances of a block.
fn launches()->Vec<(Shape,[Vec<f32>;4],Vec<f32>)> {
    let mut out=Vec::new();let none=Vec::new;
    for (i,&(rows,width,weighted)) in [(1,32,true),(1,160,true),(2,40,false),(3,300,true),(8,128,false)].iter().enumerate() {
        out.push((shape(Op::RoundedNorm,rows,width,usize::from(weighted),0,0,1.0e-6,0.0),[made(&format!("norm x {i}"),rows*width,24,-3,2),none(),none(),none()],
            if weighted {made(&format!("norm w {i}"),width,8,-1,0)} else {none()}));
    }
    // Heads of every length: under the threshold, over it, and far over it.
    for (i,&(heads,dim,scaled,factor,high)) in [(6,4,true,false,-1),(4,40,true,true,0),(2,36,false,false,2),(3,16,false,true,-4),(5,130,true,true,1)].iter().enumerate() {
        out.push((shape(Op::HeadClip,heads,dim,usize::from(scaled),0,usize::from(factor),0.0,1.5),[made(&format!("clip x {i}"),heads*dim,24,-5,high),if factor {vec![1.173_286_8]} else {none()},none(),none()],
            if scaled {made(&format!("clip w {i}"),heads*dim,8,-2,0)} else {none()}));
    }
    for (i,&(heads,dim,extent)) in [(6,4,6),(4,6,5),(3,40,70)].iter().enumerate() {
        out.push((shape(Op::RelBias,heads,dim,0,0,extent,0.0,0.0),[made(&format!("rel x {i}"),heads*dim,24,-3,0),none(),none(),none()],made(&format!("rel w {i}"),dim*extent,8,-3,0)));
    }
    for (i,&(heads,dim,taps)) in [(2,8,4),(1,32,3),(3,6,1)].iter().enumerate() {
        for form in [0,1,2,3,6,7] {for slot in 0..taps {
            let width=heads*dim;
            out.push((shape(Op::ShortConv,heads,dim,taps,0,form,0.0,0.447_213_6),[made(&format!("conv ring {i}"),taps*width,24,-3,1),if form&1!=0 {made(&format!("conv made {i}"),heads*taps,24,-6,-3)} else {none()},
                if form&2!=0 {made(&format!("conv add {i}"),width,24,-2,1)} else {none()},vec![slot as f32]],made(&format!("conv w {i}"),width*taps,8,-3,-1)));
        }}
    }
    // A plane in a straight line, a ring that has gone round, a ring that is not full, and more keys than one block.
    for (i,&(heads,dim,group,rows,extent,live,slot)) in [(6,4,3,16,6,7,6),(4,8,2,4,6,4,1),(4,8,2,4,6,2,1),(2,4,1,1100,5,1100,1099),(4,36,2,3,5,3,0),(2,4,2,2100,5,2100,7)].iter().enumerate() {
        let plane=packed(&made(&format!("plane {i}"),rows*(heads/group)*dim,8,-3,0));
        let step=vec![live as f32,slot as f32];
        out.push((shape(Op::WindowScores,heads,dim,group,rows,extent,0.0,0.353_553_4),[made(&format!("query {i}"),heads*dim,24,-3,0),plane.clone(),made(&format!("bias {i}"),heads*extent,8,-3,0),step.clone()],none()));
        let scores=made(&format!("scores {i}"),heads*rows,8,-2,1);
        out.push((shape(Op::SinkSoftmax,heads,0,0,rows,0,0.0,0.0),[scores,made(&format!("sink {i}"),heads,24,-2,1),vec![live as f32],none()],none()));
        // Weights of the live keys only, as the softmax writes them.
        let mut weights=made(&format!("weights {i}"),heads*rows,8,-12,-3);
        for (j,weight) in weights.iter_mut().enumerate() {*weight=if j%rows<live {weight.abs()} else {0.0};}
        out.push((shape(Op::WindowValues,heads,dim,group,rows,0,0.0,0.0),[weights,plane,made(&format!("gate {i}"),heads*dim,24,-3,1),step],none()));
    }
    out.push((shape(Op::Softcap,0,0,0,40,0,(1.0f64/30.0) as f32,30.0),[made("cap x",40,24,-3,6),none(),none(),none()],none()));
    out
}
#[test]
fn each_arm_equals_its_hand_form() {
    // For each operation: launches that the host ran and that equal the hand form, and launches that the host refused.
    let mut seen=BTreeMap::<String,(usize,usize)>::new();
    for (s,x,w) in launches() {
        let by_hand=hand::arm(s,[&x[0],&x[1],&x[2],&x[3]],&w).unwrap().unwrap_or_else(||panic!("no hand form of {:?}",s.op));
        assert!(by_hand.iter().all(|v|v.is_finite()),"{:?}",s);
        let entry=seen.entry(format!("{:?}",s.op)).or_default();
        let given=|i:usize|(!x[i].is_empty()).then_some(&x[i]);
        match Host.operation(s,[given(0),given(1),given(2),given(3)],&w) {
            Ok(out)=>{assert_eq!(bits(&out),bits(&by_hand),"the host arm differs from its hand form: {s:?}");entry.0+=1;},
            Err(_)=>entry.1+=1,
        }
    }
    // Every arm that this seat launches has its body and takes every launch of the list.
    assert!(hand::softcap_is_ruled(),"the host arm of the logit cap divides by the cap");
    for (op,(ran,refused)) in &seen {assert!(*ran>0 && *refused==0,"{op}: {ran} launches ran and {refused} were refused");}
    assert_eq!(seen.len(),8);
    println!("launches of each arm, each equal to its hand form in every word: {:?}",seen.iter().map(|(op,v)|format!("{op} {}",v.0)).collect::<Vec<_>>());
}
/// One visit of stored layer `layer` on the hand backend, at `position`, on `state`.
#[derive(Debug)]
struct Visited {residual:Vec<f32>,launches:BTreeMap<u32,usize>}
fn visit(fixture:&Fixture,description:&engine_model::plan::description::LoopTextDescription,layer:usize,state:&mut state::VisitState<Vec<f32>>,position:u64,keep:&[f32],inputs:&Inputs<Vec<f32>>)->Result<Visited> {
    let hand=Hand::new(Mode::All);let run=Run::new(&hand,fixture,description)?;
    let visit=*description.visits.iter().find(|visit|visit.layer==layer as u64).unwrap();
    let layout=state::layouts(description,16)?[visit.slot as usize];
    let residual=half(&run,&visit,&layout,state,position,&keep.to_vec(),inputs)?;
    Ok(Visited {residual,launches:hand.launches.borrow().clone()})
}
fn clean(fixture:&Fixture,layer:usize)->(state::Layout,state::VisitState<Vec<f32>>) {
    let d=&fixture.description;let slot=d.visits.iter().find(|visit|visit.layer==layer as u64).unwrap().slot as usize;
    let layout=state::layouts(d,16).unwrap()[slot];(layout,state::clean(&Host,&[layout]).unwrap().visits.remove(0))
}
fn all(state:&state::VisitState<Vec<f32>>)->Vec<u32> {[&state.keys,&state.values].into_iter().chain(&state.rings).flat_map(|v|bits(v)).collect()}
#[test]
fn one_visit_runs_the_launches_of_its_layer_kind_and_writes_its_rows() {
    for fixture in [Fixture::small(),Fixture::wide()] {
        let d=&fixture.description;let h=d.hidden as usize;
        let keep=made("keep",h,8,-2,1);let one=made("input",h,24,-2,1);let three=[made("input q",h,24,-2,1),made("input k",h,24,-2,1),made("input v",h,24,-2,1)];
        let count=|launches:&BTreeMap<u32,usize>,op:Op|launches.get(&(op as u32)).copied().unwrap_or(0);
        for (layer,inputs,norms,products) in [(0,Inputs::Same(one.clone()),2,7),(0,Inputs::Each(three.clone()),2,7),(1,Inputs::Each(three.clone()),4,9),(1,Inputs::Same(one.clone()),4,9)] {
            let a=d.layers[layer].attention;let (layout,mut state)=clean(&fixture,layer);let position=9;
            let out=visit(&fixture,d,layer,&mut state,position,&keep,&inputs).unwrap();
            // One norm for one weight (one launch for three inputs), else three; and the norm of the values. Six products and the output
            // product; a layer that makes taps from the token has two more.
            assert_eq!((count(&out.launches,Op::RoundedNorm),count(&out.launches,Op::Project)),(norms,products),"layer {layer}");
            assert_eq!([Op::HeadClip,Op::RelBias,Op::ShortConv,Op::PackBf16,Op::WindowScores,Op::SinkSoftmax,Op::WindowValues].map(|op|count(&out.launches,op)),[3,1,3,2,1,1,1]);
            assert_eq!(out.launches.values().sum::<usize>(),norms+products+12);
            // The inputs of the three convolutions are in their rings, as the 32-bit products.
            let (xk,xv)=match &inputs {Inputs::Same(x)|Inputs::NormedSame(x)|Inputs::NormedHalfSame(x)=>(x,x),Inputs::Each([_,k,v])|Inputs::NormedEach([_,k,v])|Inputs::NormedHalfEach([_,k,v])=>(k,v)};
            let weights=|which:Weight|<Fixture as Weights<Host>>::words(&fixture,which).unwrap();
            let normed=|x:&[f32],which:Weight|hand::rounded_norm(x,Some(weights(which)),d.epsilon as f32);
            for (ring,weight,x) in [(Ring::Key,Weight::Key(layer),normed(xk,Weight::KeyNorm(layer))),(Ring::Value,Weight::Value(layer),normed(xv,Weight::ValueNorm(layer)))] {
                let product=<Fixture as Weights<Host>>::product(&fixture,&Host,weight,&x).unwrap();let at=layout.ring_slot(ring,position)*layout.kv_channels;
                assert_eq!(bits(&state.rings[ring as usize][at..at+layout.kv_channels]),bits(&product),"layer {layer} {ring:?}");
                assert!(product.iter().any(|&v|rn(v).to_bits()!=v.to_bits()),"a ring holds the product as it is");
                assert!(state.rings[ring as usize].iter().enumerate().all(|(i,v)|(at..at+layout.kv_channels).contains(&i) || v.to_bits()==0),"only the row of the token is written");
            }
            // The finished key and value are in the row of the token: BF16 values, each key head no longer than the
            // clip, each value head with a mean square of one.
            let (head,row)=(a.head_dim as usize,layout.slot(position).unwrap()*layout.row_words);
            let unpack=|plane:&[f32]|(0..layout.kv_channels).map(|j|unpacked(&plane[row..row+layout.row_words],j)).collect::<Vec<_>>();
            let (key,value)=(unpack(&state.keys),unpack(&state.values));
            for key_head in key.chunks_exact(head) {let length=key_head.iter().map(|v|v*v).sum::<f32>().sqrt();assert!(length>0.0 && length<=a.query_key_clip as f32*1.01,"a key head of length {length}");}
            for value_head in value.chunks_exact(head) {let mean=value_head.iter().map(|v|v*v).sum::<f32>()/head as f32;assert!((mean-1.0).abs()<0.02,"a value head with mean square {mean}");}
            assert!(state.keys.iter().enumerate().all(|(i,v)|(row..row+layout.row_words).contains(&i) || v.to_bits()==0));
            // Step 7. The first layer of a block gives the convolution output alone: BF16 values. Another layer adds the
            // stream that the visit read, in 32-bit, and does not round.
            let mut other=d.clone();other.layers[layer].boundary = !d.layers[layer].boundary;
            let flipped=visit(&fixture,&other,layer,&mut clean(&fixture,layer).1,position,&keep,&inputs).unwrap().residual;
            let (alone,summed)=if d.layers[layer].boundary {(&out.residual,&flipped)} else {(&flipped,&out.residual)};
            assert!(alone.iter().all(|&v|rn(v).to_bits()==v.to_bits()));
            assert_eq!(bits(summed),bits(&keep.iter().zip(alone).map(|(k,c)|k+c).collect::<Vec<_>>()));
            assert!(summed.iter().any(|&v|rn(v).to_bits()!=v.to_bits()),"the residual is not rounded");
            // The same step again on the state it left: the same words, and the state does not change.
            let before=all(&state);
            assert_eq!(bits(&visit(&fixture,d,layer,&mut state,position,&keep,&inputs).unwrap().residual),bits(&out.residual));assert_eq!(all(&state),before);
        }
    }
}
#[test]
fn a_ring_gives_the_same_words_wherever_it_starts() {
    // A window layer has no position factor, so a token's result depends on the tokens it sees and not on where
    // the chat started. A chat that starts later on a clean state first sees rows that hold no token. Once the
    // window holds only its own tokens, and then the taps after attention hold only outputs of such tokens, a chat
    // that starts at slot 0 and chats that start at other slots give the same words, though each row sits in
    // another slot.
    for fixture in [Fixture::small(),Fixture::wide()] {
        let d=&fixture.description;let h=d.hidden as usize;let full=(d.layers[0].attention.window.unwrap()-1+d.conv.kernel-1) as usize;assert!(full<9);
        let runs=[0u64,1,5,10].map(|start|{
            let (_,mut state)=clean(&fixture,0);
            (0..9).map(|i|visit(&fixture,d,0,&mut state,start+i,&made(&format!("keep {i}"),h,8,-2,1),&Inputs::Same(made(&format!("input {i}"),h,24,-2,1))).unwrap().residual).collect::<Vec<_>>()
        });
        for other in &runs[1..] {
            for i in full..9 {assert_eq!(bits(&other[i]),bits(&runs[0][i]),"token {i}");}
            assert_ne!(bits(&other[0]),bits(&runs[0][0]),"before the window is full the chats differ: one sees rows that hold no token");
        }
    }
}
#[test]
fn a_visit_refuses_what_differs_from_its_layer() {
    let fixture=Fixture::small();let d=&fixture.description;let h=d.hidden as usize;
    let (keep,input)=(made("keep",h,8,-2,1),Inputs::Same(made("input",h,24,-2,1)));
    // The state of the other layer kind.
    assert!(visit(&fixture,d,0,&mut clean(&fixture,1).1,0,&keep,&input).is_err());
    let hand=Hand::new(Mode::All);let run=Run::new(&hand,&fixture,d).unwrap();let (wrong,mut state)=clean(&fixture,1);
    assert!(half(&run,&d.visits[0],&wrong,&mut state,0,&keep,&input).unwrap_err().contains("the state layout differs"));
    // A stream or an input of another width.
    let (_,mut state)=clean(&fixture,0);
    assert!(visit(&fixture,d,0,&mut state,0,&keep[1..],&input).unwrap_err().contains("the stream differs from the width"));
    assert!(visit(&fixture,d,0,&mut state,0,&keep,&Inputs::Same(made("short",h-1,24,-2,1))).unwrap_err().contains("a norm input or weight differs"));
    assert!(all(&state).iter().all(|&word|word==0),"a refused visit wrote nothing");
    // A straight plane ends at its capacity. A ring has no end.
    let (_,mut plane)=clean(&fixture,1);
    assert!(visit(&fixture,d,1,&mut plane,16,&keep,&input).unwrap_err().contains("longer than the state's capacity"));
    assert!(visit(&fixture,d,0,&mut state,1_000_000,&keep,&input).is_ok());
    // A layer with no size of the relative term, or with an odd head, is not built.
    let mut odd=d.clone();odd.layers[0].attention.relative.extent=0;
    assert!(visit(&fixture,&odd,0,&mut clean(&fixture,0).1,0,&keep,&input).unwrap_err().contains("the attention shape is not built"));
}
