//! Host tests of the arms in `block_attention.rs`: small shapes with hand-worked values.
use crate::{block_attention::{distance_sum,distance_tree,rel_part,softmax_part,BLOCK},block_ops::Terms,causal_ops::{reference,Op,Shape},fixed_math::{exp,rn}};

fn shape(op:Op)->Shape {Shape {op,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}}
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
/// Two BF16 values a word, as `PackBf16` writes them: the first in the low half, the second in the high half.
fn pack(values:&[f32])->Vec<f32> {
    values.chunks(2).map(|pair|{
        assert!(pair.iter().all(|v|rn(*v).to_bits()==v.to_bits()),"a packed value must be a BF16 value");
        f32::from_bits((pair[0].to_bits()>>16)|(pair[1].to_bits()&0xffff_0000))
    }).collect()
}

#[test] fn head_clip_leaves_a_short_head_and_scales_a_long_head_to_the_threshold() {
    let clip=|bound:f32|Shape {heads:2,dim:2,bound,..shape(Op::HeadClip)};
    // Head 0 is (3, 4): length 5. Head 1 is zero. Threshold 10: both stay.
    assert_eq!(reference(clip(10.0),[&[3.0,4.0,0.0,0.0],&[],&[],&[]],&[]).unwrap(),[3.0,4.0,0.0,0.0]);
    // Threshold 2.5: the factor is 2.5 / 5.000001, a little under 0.5. 3 and 4 become 1.4999997 and 1.9999996,
    // and the BF16 round gives 1.5 and 2. Head 1 is (0.5, 0): length 0.5, it stays.
    assert_eq!(reference(clip(2.5),[&[3.0,4.0,0.5,0.0],&[],&[],&[]],&[]).unwrap(),[1.5,2.0,0.5,0.0]);
    // A head at the threshold: the epsilon makes the factor 0.9999998, and the round gives the head back.
    assert_eq!(reference(clip(5.0),[&[3.0,4.0,0.0,5.0],&[],&[],&[]],&[]).unwrap(),[3.0,4.0,0.0,5.0]);
    // The input is read as BF16: 3.0000002 is read as 3.
    assert_eq!(reference(clip(2.5),[&[f32::from_bits(0x4040_0001),4.0,0.5,0.0],&[],&[],&[]],&[]).unwrap(),[1.5,2.0,0.5,0.0]);
}
#[test] fn head_clip_applies_the_scale_before_the_length_and_the_factor_after_the_round() {
    let s=Shape {heads:1,dim:2,taps:1,bound:2.5,..shape(Op::HeadClip)};
    // (6, 8) with the scale (0.5, 0.5) is (3, 4): as above.
    assert_eq!(reference(s,[&[6.0,8.0],&[],&[],&[]],&[0.5,0.5]).unwrap(),[1.5,2.0]);
    // With the factor 1.5 after the clip: 2.25 and 3.
    assert_eq!(reference(Shape {cols:1,..s},[&[6.0,8.0],&[1.5],&[],&[]],&[0.5,0.5]).unwrap(),[2.25,3.0]);
    // The factor meets the ROUNDED clip result. With the factor 1 + 3 / 256: 2 * 1.01171875 = 2.0234375 is a tie in
    // BF16 and goes to the even neighbor 2.03125. Without the first round, 1.9999996 * 1.01171875 is under the tie
    // and would go to 2.015625. The other value: 1.5 * 1.01171875 = 1.517578 goes to 1.515625.
    assert_eq!(reference(Shape {cols:1,..s},[&[6.0,8.0],&[1.0+3.0/256.0],&[],&[]],&[0.5,0.5]).unwrap(),[1.515_625,2.031_25]);
}
#[test] fn the_clip_factor_is_a_reciprocal_and_then_a_product() {
    // The reference makes a number over a tensor in two rounded steps. Find heads where one division gives another
    // BF16 word, and hold the arm to the two-step form there.
    let (mut found,mut seed)=(0,1u32);
    for _ in 0..8_000_000 {
        seed=seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let (a,b)=(rn((seed>>8) as f32/65_536.0+1.0),rn((seed&0xff) as f32/16.0+0.5));
        let threshold=1.25f32;
        let length=(a*a+b*b).sqrt();
        let two_steps=threshold*(1.0/(length+1e-6));
        let division=threshold/(length+1e-6);
        if two_steps>=1.0 || rn(a*two_steps).to_bits()==rn(a*division).to_bits() {continue;}
        found+=1;
        let got=reference(Shape {heads:1,dim:2,bound:threshold,..shape(Op::HeadClip)},[&[a,b],&[],&[],&[]],&[]).unwrap();
        assert_eq!(bits(&got),bits(&[rn(a*two_steps),rn(b*two_steps)]),"head ({a}, {b})");
        if found==4 {break;}
    }
    assert!(found>0,"no head tells the two forms apart");
}
#[test] fn rel_bias_is_a_chain_over_the_state_rounded_once() {
    let s=Shape {heads:2,dim:2,cols:3,..shape(Op::RelBias)};
    // Table rows: j = 0 is (1, 2, 3), j = 1 is (0.5, 0.25, -1). Head 0 is (1, 2), head 1 is (-1, 0.5).
    let table=[1.0,2.0,3.0,0.5,0.25,-1.0];
    assert_eq!(reference(s,[&[1.0,2.0,-1.0,0.5],&[],&[],&[]],&table).unwrap(),[2.0,2.5,1.0,-0.75,-1.875,-3.5]);
    // 1 + 2^-9 is no BF16 value: the sum is rounded to 1.
    let one=Shape {heads:1,dim:2,cols:1,..shape(Op::RelBias)};
    assert_eq!(reference(one,[&[1.0,1.0],&[],&[],&[]],&[1.0,0.001_953_125]).unwrap(),[1.0]);
}
/// Review REVARM4: the units of 8 distances (an extent that is a multiple of 8) and of 1 give, unit by unit as the
/// entries write them, the words of the definition (one chain over the state from +0.0 in rising j, rounded once),
/// written here on its own; with zeros of both signs, NaN and infinities in the state.
#[test] fn rel_bias_units_of_eight_and_of_one_give_the_words_of_the_definition() {
    assert_eq!((rel_part(1),rel_part(7),rel_part(8),rel_part(9),rel_part(16),rel_part(52),rel_part(1024)),(1,1,8,1,8,1,8));
    let mut state=0x1f83_d9abu32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    for (heads,dim,extent) in [(24usize,16usize,1024usize),(3,2,8),(2,5,16),(1,1,9),(2,16,52),(4,3,7),(1,2,1)] {
        let s=Shape {heads:heads as u32,dim:dim as u32,cols:extent as u32,..shape(Op::RelBias)};
        let x:Vec<f32>=(0..heads*dim).map(|i|match i%13 {3=>-0.0,5=>f32::NAN,7=>f32::NEG_INFINITY,_=>((next()>>8) as f32/8_388_608.0-1.0)*3.0}).collect();
        let w:Vec<f32>=(0..dim*extent).map(|i|if i%11==4 {0.0} else {((next()>>8) as f32/8_388_608.0-1.0)*2.0}).collect();
        let mut definition=vec![0.0f32;heads*extent];
        for head in 0..heads {for dist in 0..extent {
            let mut sum=0.0f32;
            for j in 0..dim {sum+=rn(x[head*dim+j])*w[j*extent+dist];}
            definition[head*extent+dist]=rn(sum);
        }}
        let (_,_,units,words)=crate::block_attention::lengths(s).unwrap();
        assert_eq!((units,words),(heads*extent/rel_part(extent),rel_part(extent)));
        let mut out=vec![f32::from_bits(0x7fa0_beef);units*words];
        for index in 0..units {crate::block_attention::unit(s,index,[&x,&[],&[],&[]],&w,&mut out[index*words..(index+1)*words]);}
        assert_eq!(bits(&out),bits(&definition),"heads {heads} dim {dim} extent {extent}");
    }
}
/// Review REVARM4: the shapes RelBias refuses, on the host and in a rows launch.
#[test] fn rel_bias_refuses_the_shapes_it_does_not_hold() {
    use crate::block_attention::lengths;
    let good=Shape {heads:24,dim:16,cols:1024,..shape(Op::RelBias)};
    assert!(lengths(good).is_some());
    for s in [Shape {heads:0,..good},Shape {dim:0,..good},Shape {cols:0,..good},Shape {rows:1,..good},Shape {taps:1,..good},
        Shape {bound:1.0,..good},Shape {bound:f32::NAN,..good},Shape {epsilon:1.0e-6,..good},Shape {epsilon:f32::INFINITY,..good}] {
        assert_eq!(lengths(s),None,"{s:?}");
    }
    // Counts past what the device ABI holds are refused by the checked counts of both entries.
    for s in [Shape {heads:u32::MAX,dim:2,cols:8,..good},Shape {heads:65_536,dim:1,cols:u32::MAX-7,..good},Shape {heads:1,dim:u32::MAX,cols:8,..good}] {
        assert_eq!(crate::block_ops::checked_lengths(s),None,"{s:?}");
    }
    // A rows launch: no row, too many rows, a mask bit of an input the arm does not read.
    use crate::block_rows::{lengths as rows,ROWS_MAX};
    assert!(rows(good,4,0).is_some() && rows(good,4,1).is_some());
    for (count,mask) in [(0usize,0u32),(ROWS_MAX+1,0),(4,2),(4,4),(4,8)] {assert_eq!(rows(good,count,mask),None,"rows {count} mask {mask}");}
}
/// Keys of the score tests: 2 query heads on 1 key head, 2 values a head, 4 rows.
/// The live keys by distance are (1, 1), (2, 0), (0.5, -1). The fourth row is not live and holds 9.
fn score_case(slot:usize)->(Shape,Vec<f32>,[f32;2]) {
    let by_distance=[[1.0,1.0],[2.0,0.0],[0.5,-1.0]];
    let mut rows=[[9.0f32,9.0];4];
    for (dist,key) in by_distance.iter().enumerate() {rows[(slot+4-dist)%4] = *key;}
    (Shape {heads:2,dim:2,taps:2,rows:4,cols:2,bound:0.5,..shape(Op::WindowScores)},pack(&rows.concat()),[3.0,slot as f32])
}
#[test] fn window_scores_follow_the_distance_wherever_the_ring_starts() {
    let query=[1.0,2.0,-1.0,0.5];
    let bias=[0.25,-0.5,1.0,0.0];
    // Head 0: 3 * 0.5 + 0.25, 2 * 0.5 - 0.5, -1.5 * 0.5 + 0 (distance 2 is outside the bias). Head 1: -0.5 * 0.5 + 1,
    // -2 * 0.5 + 0, -1 * 0.5. The slot that is not live gives 0.
    let want=[1.75,0.5,-0.75,0.0,0.75,-1.0,-0.5,0.0];
    for slot in [1,3,2,0] {
        let (s,keys,step)=score_case(slot);
        assert_eq!(bits(&reference(s,[&query,&keys,&bias,&step],&[]).unwrap()),bits(&want),"slot {slot}");
    }
}
#[test] fn window_scores_round_after_the_dot_after_the_scale_and_after_the_bias() {
    // Dot 2.2578125 is a tie in BF16 and goes to 2.25. Half of it is 1.125. With the bias 2^-8 it is a tie again
    // and stays 1.125. One round at the end would give 1.1328125.
    let s=Shape {heads:1,dim:2,taps:1,rows:1,cols:1,bound:0.5,..shape(Op::WindowScores)};
    let got=reference(s,[&[1.5,1.0],&pack(&[1.5,0.007_812_5]),&[0.003_906_25],&[1.0,0.0]],&[]).unwrap();
    assert_eq!(got,[1.125]);
}
#[test] fn window_scores_take_the_key_head_of_each_query_head() {
    // 2 query heads, 2 key heads (group 1): head 0 meets (1, 0), head 1 meets (0, 1).
    let s=Shape {heads:2,dim:2,taps:1,rows:1,cols:1,bound:1.0,..shape(Op::WindowScores)};
    let got=reference(s,[&[3.0,5.0,7.0,11.0],&pack(&[1.0,0.0,0.0,1.0]),&[0.0,0.0],&[1.0,0.0]],&[]).unwrap();
    assert_eq!(got,[3.0,11.0]);
}
#[test] fn a_step_word_outside_the_rows_is_refused() {
    let query=[1.0,2.0,-1.0,0.5];
    let bias=[0.25,-0.5,1.0,0.0];
    let (s,keys,_)=score_case(1);
    for step in [[5.0,1.0],[3.0,4.0],[1.5,1.0],[3.0,-1.0],[3.0,0.5]] {
        assert!(reference(s,[&query,&keys,&bias,&step],&[]).is_err(),"step {step:?}");
    }
    let softmax=Shape {heads:1,rows:2,..shape(Op::SinkSoftmax)};
    for count in [3.0,0.5,-1.0] {assert!(reference(softmax,[&[0.0,0.0],&[0.0],&[count],&[]],&[]).is_err());}
    let values=Shape {heads:1,dim:2,taps:1,rows:2,..shape(Op::WindowValues)};
    assert!(reference(values,[&[1.0,0.0],&pack(&[1.0,1.0,1.0,1.0]),&[0.0,0.0],&[3.0,0.0]],&[]).is_err());
    assert!(reference(values,[&[1.0,0.0],&pack(&[1.0,1.0,1.0,1.0]),&[0.0,0.0],&[1.0,2.0]],&[]).is_err());
}
#[test] fn sink_softmax_shares_the_weight_with_the_sink() {
    let s=|heads,rows|Shape {heads,rows,..shape(Op::SinkSoftmax)};
    // Four equal scores and an equal sink: each weight is 1 / 5, and BF16 keeps 0.20019531.
    let got=reference(s(1,4),[&[5.0;4],&[5.0],&[4.0],&[]],&[]).unwrap();
    assert_eq!(bits(&got),[0x3e4d_0000;4]);
    // Two heads, two live keys of four: each weight is 1 / 3, BF16 0.33398438. A slot that is not live gives 0.
    let got=reference(s(2,4),[&[0.0,0.0,7.0,7.0,1.0,1.0,9.0,9.0],&[0.0,1.0],&[2.0],&[]],&[]).unwrap();
    assert_eq!(bits(&got),[0x3eab_0000,0x3eab_0000,0,0,0x3eab_0000,0x3eab_0000,0,0]);
    // A sink far below takes nothing. A sink far above takes all but a trace.
    assert_eq!(reference(s(1,1),[&[3.0],&[-200.0],&[1.0],&[]],&[]).unwrap(),[1.0]);
    let got=reference(s(1,1),[&[0.0],&[100.0],&[1.0],&[]],&[]).unwrap();
    assert!(got[0]>0.0 && got[0]<1.0e-37);
    // No live key: every weight is 0.
    assert_eq!(reference(s(1,3),[&[4.0,4.0,4.0],&[0.0],&[0.0],&[]],&[]).unwrap(),[0.0;3]);
}
/// The softmax with a sink, written from its definition with every lane and every block sum kept in a list.
fn softmax_by_definition(scores:&[f32],sink:f32,n:usize)->Vec<f32> {
    let top=scores[..n].iter().copied().fold(sink,f32::max);
    let terms=scores[..n].iter().map(|&score|exp(score-top)).collect::<Vec<_>>();
    let blocks=terms.chunks(BLOCK).map(|block|{
        let mut lanes=vec![0.0f32;32];
        for (dist,term) in block.iter().enumerate() {lanes[dist%32]+=term;}
        for distance in [16,8,4,2,1] {let old=lanes.clone();for lane in 0..32 {lanes[lane]=old[lane]+old[lane^distance];}}
        lanes[0]
    }).collect::<Vec<_>>();
    let mut total=0.0f32;
    for block in blocks {total+=block;}
    total+=exp(sink-top);
    scores.iter().enumerate().map(|(dist,_)|if dist<n {rn(terms[dist]/total)} else {0.0}).collect()
}
#[test] fn sink_softmax_adds_in_blocks_of_lanes_by_distance_and_the_sink_last() {
    let rows=2560;
    let scores=(0..rows).map(|i|rn(((i*7919%4001) as f32-2000.0)/250.0)).collect::<Vec<_>>();
    for n in [1,31,32,33,1023,1024,1025,2500] {
        let got=reference(Shape {heads:1,rows:rows as u32,..shape(Op::SinkSoftmax)},[&scores,&[1.5],&[n as f32],&[]],&[]).unwrap();
        assert_eq!(bits(&got),bits(&softmax_by_definition(&scores,1.5,n)),"live keys {n}");
        let share=got.iter().sum::<f32>();
        assert!(share>0.0 && share<1.01,"the keys and the sink share one unit of weight; keys {share}");
    }
}
/// Values of the value tests: as `score_case`. The live values by distance are (2, 4), (4, -8), (8, 0).
fn value_case(slot:usize)->(Shape,Vec<f32>,[f32;2]) {
    let by_distance=[[2.0,4.0],[4.0,-8.0],[8.0,0.0]];
    let mut rows=[[9.0f32,9.0];4];
    for (dist,value) in by_distance.iter().enumerate() {rows[(slot+4-dist)%4] = *value;}
    (Shape {heads:2,dim:2,taps:2,rows:4,..shape(Op::WindowValues)},pack(&rows.concat()),[3.0,slot as f32])
}
#[test] fn window_values_follow_the_distance_and_apply_the_gate() {
    // Head 0 weighs the three values 0.5, 0.25, 0.25: (4, 0). Head 1 takes the first value: (2, 4).
    let weights=[0.5,0.25,0.25,0.0,1.0,0.0,0.0,0.0];
    for slot in [1,3,2,0] {
        let (s,values,step)=value_case(slot);
        // A gate of 0 is a factor of one half.
        assert_eq!(reference(s,[&weights,&values,&[0.0;4],&step],&[]).unwrap(),[2.0,0.0,1.0,2.0],"slot {slot}");
        // A gate of 40 is a factor of 1, and a gate that is read as BF16: 40.000004 is 40.
        let gate=[40.0,40.0,f32::from_bits(0x4220_0001),40.0];
        assert_eq!(reference(s,[&weights,&values,&gate,&step],&[]).unwrap(),[4.0,0.0,2.0,4.0],"slot {slot}");
    }
}
/// One head, one key head, 2 values a head, all weights 1, the gate open. `terms` gives channel 0 by distance.
fn value_sum(rows:usize,slot:usize,terms:&[(usize,f32)],n:usize)->f32 {
    let mut plane=vec![0.0f32;rows*2];
    for &(dist,value) in terms {plane[(slot+rows-dist)%rows*2]=value;}
    let s=Shape {heads:1,dim:2,taps:1,rows:rows as u32,..shape(Op::WindowValues)};
    reference(s,[&vec![1.0;rows],&pack(&plane),&[40.0,40.0],&[n as f32,slot as f32]],&[]).unwrap()[0]
}
#[test] fn a_sum_over_keys_has_32_lanes_by_distance_in_blocks_of_1024() {
    let big=134_217_728.0f32;
    // Lanes: distance 0 and distance 32 share lane 0, so the 1 is lost beside 2^27. Lane 1 holds -2^27 and the
    // join gives 0. A chain in rising distance would give 1.
    for slot in [0,40,63] {assert_eq!(value_sum(64,slot,&[(0,big),(1,-big),(32,1.0)],33),0.0,"slot {slot}");}
    // The 1 in its own lane. Lane 0: 2^27. Lane 1: -2^27. Lane 2: 1. Join: lane 0 takes lane 2 first (the 1 is
    // lost), then lane 1: 0. A chain in rising distance would give 1.
    assert_eq!(value_sum(64,7,&[(0,big),(1,-big),(2,1.0)],33),0.0);
    // Blocks: distance 0 is block 0; distances 1024 and 1025 are block 1, in lanes 0 and 1. Block 1 is -2^27 + 1,
    // which is -2^27, and the two block sums give 0. With no blocks, lane 0 would hold 2^27 - 2^27 and the 1 stays.
    for slot in [5,2047] {assert_eq!(value_sum(2048,slot,&[(0,big),(1024,-big),(1025,1.0)],1026),0.0,"slot {slot}");}
    // The block sums are a chain in rising block number: 1 + 2^27 is 2^27, then -2^27 gives 0. A join of block 1
    // with block 2 first would keep the 1.
    assert_eq!(value_sum(4096,9,&[(0,1.0),(1024,big),(2048,-big)],2049),0.0);
}
#[test] fn the_arms_take_the_shapes_they_need_and_refuse_the_others() {
    let lengths=|s:Shape|s.lengths();
    // Sizes of one model, as an example of a large shape: 24 query heads of 128 on 8 key heads, a window of 1024.
    let scores=Shape {heads:24,dim:128,taps:3,rows:1024,cols:1024,bound:0.088_388_35,..shape(Op::WindowScores)};
    assert_eq!(lengths(scores),Some(([3072,524_288,24_576,2],0,24_576,1)));
    assert_eq!(lengths(Shape {heads:24,rows:1024,..shape(Op::SinkSoftmax)}),Some(([24_576,24,1,0],0,24*32,32)));
    // A head of the softmax is written in parts of 32 words when its rows are a multiple of 32, else in one unit.
    assert_eq!(lengths(Shape {heads:3,rows:96,..shape(Op::SinkSoftmax)}),Some(([288,3,1,0],0,9,32)));
    assert_eq!(lengths(Shape {heads:3,rows:33,..shape(Op::SinkSoftmax)}),Some(([99,3,1,0],0,3,33)));
    assert_eq!((softmax_part(1024),softmax_part(256),softmax_part(32),softmax_part(31),softmax_part(1),softmax_part(100)),(32,32,32,31,1,100));
    assert_eq!(lengths(Shape {cols:0,bound:0.0,..Shape {op:Op::WindowValues,..scores}}),Some(([24_576,524_288,3072,2],0,3072,1)));
    assert_eq!(lengths(Shape {heads:24,dim:16,cols:1024,..shape(Op::RelBias)}),Some(([384,0,0,0],16_384,24*128,8)));
    assert_eq!(lengths(Shape {heads:2,dim:2,cols:3,..shape(Op::RelBias)}),Some(([4,0,0,0],6,6,1)));
    assert_eq!((rel_part(1024),rel_part(8),rel_part(52),rel_part(3)),(8,8,1,1));
    assert_eq!(lengths(Shape {heads:24,dim:128,taps:1,cols:1,bound:8.0,..shape(Op::HeadClip)}),Some(([3072,1,0,0],3072,24*4,32)));
    assert_eq!(lengths(Shape {heads:8,dim:128,bound:8.0,..shape(Op::HeadClip)}),Some(([1024,0,0,0],0,8*4,32)));
    // A head of the clip is written in parts of 32 words when its width is a multiple of 32, else in one unit.
    assert_eq!(lengths(Shape {heads:24,dim:16,taps:1,cols:1,bound:8.0,..shape(Op::HeadClip)}),Some(([384,1,0,0],384,24,16)));
    assert_eq!(lengths(Shape {heads:3,dim:96,bound:8.0,..shape(Op::HeadClip)}),Some(([288,0,0,0],0,9,32)));
    // Refused: an odd head width, a group that does not divide the heads, no threshold, a field the arm does not use.
    for s in [Shape {dim:127,..scores},Shape {taps:5,..scores},Shape {bound:0.0,..scores},Shape {epsilon:1.0,..scores},
        Shape {heads:2,dim:2,..shape(Op::HeadClip)},Shape {heads:2,dim:2,taps:2,bound:1.0,..shape(Op::HeadClip)},
        Shape {heads:2,dim:2,rows:1,bound:1.0,..shape(Op::HeadClip)},Shape {heads:1,dim:2,rows:4,..shape(Op::SinkSoftmax)},
        Shape {heads:1,dim:2,cols:3,taps:1,..shape(Op::RelBias)},Shape {cols:4,..Shape {op:Op::WindowValues,bound:0.0,..scores}}] {
        assert_eq!(lengths(s),None,"{s:?}");
    }
    // A buffer one word short is refused by the host form.
    let (s,keys,step)=score_case(1);
    assert!(reference(s,[&[1.0,2.0,-1.0],&keys,&[0.25,-0.5,1.0,0.0],&step],&[]).is_err());
}
/// Terms held in a list, for the two forms of the sum over keys.
struct Held<'a>(&'a [f32]);
impl Terms for Held<'_> {fn term(&self,dist:usize)->f32 {self.0[dist]}}
#[test] fn the_register_form_of_the_sum_over_keys_gives_the_bits_of_its_definition() {
    // Terms of many sizes and both signs, so another order of the sums gives another last bit.
    let mut state=0x9e37_79b9u32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    for n in [0usize,1,2,31,32,33,63,64,65,1000,1023,1024,1025,1056,2047,2048,2049,2500,3072,4097] {
        for round in 0..12 {
            let terms=(0..n).map(|_|{let bits=next();let value=((bits>>8) as f32/8_388_608.0-1.0)*[1.0e-3,1.0,3.0e3,7.0e7][(bits&3) as usize];
                if round%3==0 && bits%5==0 {0.0} else {value}}).collect::<Vec<_>>();
            assert_eq!(distance_tree(n,&Held(&terms)).to_bits(),distance_sum(n,|dist|terms[dist]).to_bits(),"terms {n} round {round}");
        }
    }
    // The hand-worked lanes of the test above, on the form itself: the 1 is lost beside 2^27 in lane 0.
    let big=134_217_728.0f32;
    let mut terms=vec![0.0f32;33];terms[0]=big;terms[1]=0.0-big;terms[32]=1.0;
    assert_eq!(distance_tree(33,&Held(&terms)),0.0);
    // A chain in rising distance would give 1.
    assert_eq!(terms.iter().sum::<f32>(),1.0);
}
#[test] fn sink_softmax_in_parts_gives_the_words_of_the_whole_head() {
    // Three heads, each with its own scores and sink. Rows that are a multiple of 32 (parts) and rows that are not.
    for rows in [32usize,96,100,1024,1056,2080] {
        let heads=3usize;
        let scores=(0..heads*rows).map(|i|rn(((i*7919%4001) as f32-2000.0)/250.0)).collect::<Vec<_>>();
        let sinks=[1.5f32,-3.0,8.0];
        for n in [0,1,31,32,33,rows-1,rows] {
            if n>rows {continue;}
            let got=reference(Shape {heads:heads as u32,rows:rows as u32,..shape(Op::SinkSoftmax)},[&scores,&sinks,&[n as f32],&[]],&[]).unwrap();
            let want=(0..heads).flat_map(|head|softmax_by_definition(&scores[head*rows..(head+1)*rows],sinks[head],n)).collect::<Vec<_>>();
            assert_eq!(bits(&got),bits(&want),"rows {rows} live keys {n}");
        }
    }
}
/// `WindowValues` written from its definition: the row by a remainder, the terms in a list, the sum by `distance_sum`.
fn values_by_definition(s:Shape,weights:&[f32],plane:&[f32],gate:&[f32],n:usize,slot:usize)->Vec<f32> {
    let (h,d,t,r)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize);
    (0..h*d).map(|index|{
        let (head,channel)=(index/d,index%d);
        let terms=(0..n).map(|dist|{
            let word=plane[((slot+r-dist)%r*(h/t)+head/t)*(d/2)+channel/2].to_bits();
            weights[head*r+dist]*f32::from_bits(if channel%2==0 {word<<16} else {word&0xffff_0000})
        }).collect::<Vec<_>>();
        rn(rn(distance_sum(n,|dist|terms[dist]))*crate::fixed_math::sigmoid(rn(gate[head*d+channel])))
    }).collect()
}
#[test] fn window_values_give_the_words_of_the_definition_at_every_slot() {
    let mut state=0x1234_5677u32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    for (heads,dim,group,rows) in [(6u32,4u32,3u32,40usize),(4,6,2,1056),(2,2,1,2100),(3,8,3,33)] {
        let s=Shape {heads,dim,taps:group,rows:rows as u32,..shape(Op::WindowValues)};
        let (h,d)=(heads as usize,dim as usize);
        let weights=(0..h*rows).map(|_|rn((next()>>8) as f32/16_777_216.0)).collect::<Vec<_>>();
        let plane=(0..rows*(h/group as usize)*d/2).map(|_|{let bits=next();f32::from_bits((bits&0x3fff_3fff)|0x3c00_3c00|((bits&1)<<15)|((bits&2)<<30))}).collect::<Vec<_>>();
        let gate=(0..h*d).map(|_|((next()>>8) as f32/8_388_608.0-1.0)*4.0).collect::<Vec<_>>();
        for (n,slot) in [(rows,rows-1),(rows,0),(rows,rows/2),(rows-1,7),(1,5),(0,3),(33,2),(33,rows-1)] {
            let got=reference(s,[&weights,&plane,&gate,&[n as f32,slot as f32]],&[]).unwrap();
            assert_eq!(bits(&got),bits(&values_by_definition(s,&weights,&plane,&gate,n,slot)),"{heads} heads rows {rows} live {n} slot {slot}");
        }
    }
}
/// `WindowScores` written from its definition: the row by a remainder, one value of the key at a time.
fn scores_by_definition(s:Shape,query:&[f32],plane:&[f32],bias:&[f32],n:usize,slot:usize)->Vec<f32> {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    (0..h*r).map(|index|{
        let (head,dist)=(index/r,index%r);
        if dist>=n {return 0.0;}
        let mut sum=0.0f32;
        for j in 0..d {
            let word=plane[((slot+r-dist)%r*(h/t)+head/t)*(d/2)+j/2].to_bits();
            sum+=rn(query[head*d+j])*f32::from_bits(if j%2==0 {word<<16} else {word&0xffff_0000});
        }
        rn(rn(rn(sum)*s.bound)+if dist<k {bias[head*k+dist]} else {0.0})
    }).collect()
}
#[test] fn window_scores_give_the_words_of_the_definition_at_every_slot() {
    let mut state=0x0bad_5eedu32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    for (heads,dim,group,rows,extent) in [(6u32,4u32,3u32,40usize,16u32),(4,6,2,300,300),(2,2,1,65,80),(3,128,3,33,8)] {
        let s=Shape {heads,dim,taps:group,rows:rows as u32,cols:extent,bound:0.088_388_35,..shape(Op::WindowScores)};
        let (h,d)=(heads as usize,dim as usize);
        // The query holds FP32 values that are no BF16 values: the arm reads them through the round.
        let query=(0..h*d).map(|_|((next()>>8) as f32/8_388_608.0-1.0)*2.0).collect::<Vec<_>>();
        let plane=(0..rows*(h/group as usize)*d/2).map(|_|{let bits=next();f32::from_bits((bits&0x3fff_3fff)|0x3c00_3c00|((bits&1)<<15)|((bits&2)<<30))}).collect::<Vec<_>>();
        let bias=(0..h*extent as usize).map(|_|rn(((next()>>8) as f32/8_388_608.0-1.0)*3.0)).collect::<Vec<_>>();
        for (n,slot) in [(rows,rows-1),(rows,0),(rows,rows/2),(rows-1,7),(1,5),(0,3),(33,2),(33,rows-1)] {
            let got=reference(s,[&query,&plane,&bias,&[n as f32,slot as f32]],&[]).unwrap();
            assert_eq!(bits(&got),bits(&scores_by_definition(s,&query,&plane,&bias,n,slot)),"{heads} heads rows {rows} live {n} slot {slot}");
        }
    }
}
/// `HeadClip` written from its definition: one head at a time, the squares on `causal_ops::square_sum`.
fn clip_by_definition(s:Shape,x:&[f32],position:Option<f32>,scale:&[f32])->Vec<f32> {
    let d=s.dim as usize;
    x.chunks_exact(d).enumerate().flat_map(|(head,row)|{
        let values=row.iter().enumerate().map(|(j,&value)|if s.taps==1 {rn(value)*scale[head*d+j]} else {rn(value)}).collect::<Vec<_>>();
        let reciprocal=1.0/(crate::causal_ops::square_sum(&values).sqrt()+1e-6);
        let factor=if s.bound*reciprocal>1.0 {1.0} else {s.bound*reciprocal};
        values.iter().map(|&value|{let clipped=rn(value*factor);match position {Some(p)=>rn(clipped*p),None=>clipped}}).collect::<Vec<_>>()
    }).collect()
}
#[test] fn head_clip_in_parts_gives_the_words_of_the_whole_head() {
    let mut state=0x6d2b_79f5u32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    // Widths that are a multiple of 32 (parts) and widths that are not; heads over and under the threshold.
    for (heads,dim) in [(24u32,128u32),(8,128),(24,16),(3,96),(5,33),(2,1),(4,64),(3,160)] {
        for (scaled,placed,bound) in [(true,true,1.0f32),(false,false,8.0),(true,false,0.25),(false,true,3.0)] {
            let s=Shape {heads,dim,taps:u32::from(scaled),cols:u32::from(placed),bound,..shape(Op::HeadClip)};
            let count=(heads*dim) as usize;
            // The input holds FP32 values that are no BF16 values: the arm reads them through the round.
            let x=(0..count).map(|_|((next()>>8) as f32/8_388_608.0-1.0)*0.4).collect::<Vec<_>>();
            let scale=if scaled {(0..count).map(|_|(next()>>8) as f32/8_388_608.0+0.25).collect::<Vec<_>>()} else {Vec::new()};
            let position=if placed {vec![1.0+3.0/256.0]} else {Vec::new()};
            let got=reference(s,[&x,&position,&[],&[]],&scale).unwrap();
            assert_eq!(bits(&got),bits(&clip_by_definition(s,&x,position.first().copied(),&scale)),"{heads} heads of {dim} scale {scaled} factor {placed} bound {bound}");
        }
    }
}

/// A few odd words among the values of a staged-form test: signed zeros, an infinity, a NaN, a large value.
fn odd(seed:u32,value:f32)->f32 {
    match seed%61 {0=>0.0,1=>-0.0,2=>f32::NEG_INFINITY,3=>f32::NAN,4=>40.0,5=>f32::INFINITY,_=>value}
}
/// The staged `SinkSoftmax` of the GPU entry, phase by phase as a block of SUM_THREADS threads runs it.
fn softmax_staged(s:Shape,x:[&[f32];4])->Vec<f32> {
    use crate::block_attention::{softmax_add,softmax_best,softmax_exp,softmax_live,softmax_part_sum,softmax_pick,softmax_sink,softmax_total,softmax_word,SOFTMAX_GROUP};
    use crate::block_ops::{APART,SUM_THREADS};
    let (h,r)=(s.heads as usize,s.rows as usize);
    let mut out=vec![0.0f32;h*r];
    for (head,words) in out.chunks_exact_mut(r).enumerate() {
        let Some(n)=softmax_live(s,x) else {words.fill(f32::NAN);continue;};
        let scores=&x[0][head*r..(head+1)*r];
        let sink=softmax_sink(head,x);
        let bests=(0..SUM_THREADS).map(|t|softmax_best(n,t,SUM_THREADS,&|d|scores[d])).collect::<Vec<_>>();
        let top=softmax_pick(sink,bests.into_iter());
        let e=|d:usize|softmax_exp(scores[d],top);
        let blocks=n.div_ceil(BLOCK);
        let (mut chain,mut group)=(0.0f32,0);
        while group<blocks {
            let parts=(0..SUM_THREADS).map(|t|{let q=group+t/APART;if q<blocks {let first=q*BLOCK;softmax_part_sum(first,(first+BLOCK).min(n),t%APART,&e)} else {f32::NAN}}).collect::<Vec<_>>();
            let mut k=0;
            while k<SOFTMAX_GROUP && group+k<blocks {chain=softmax_add(chain,core::array::from_fn(|p|parts[k*APART+p]));k+=1;}
            group+=SOFTMAX_GROUP;
        }
        let total=softmax_total(chain,sink,top);
        for (j,word) in words.iter_mut().enumerate() {*word=softmax_word(j,n,if j<n {e(j)} else {0.0},total);}
    }
    out
}
/// The staged `WindowValues` of the GPU entry: the parts of each block of distances, joined, the block sums chained.
fn values_staged(s:Shape,x:[&[f32];4])->Vec<f32> {
    use crate::block_attention::{values_live,values_part,values_word};
    (0..(s.heads*s.dim) as usize).map(|index|match values_live(s,x) {
        None=>f32::NAN,
        Some((n,slot))=>{
            let (mut total,mut first)=(0.0f32,0);
            while first<n {
                let end=if n-first>BLOCK {first+BLOCK} else {n};
                total+=crate::block_ops::join_parts(core::array::from_fn(|p|values_part(s,index,slot,x,first,end,p)));
                first=end;
            }
            values_word(s,index,x,total)
        },
    }).collect()
}
/// The words of `unit` for every unit of a launch.
fn by_units(s:Shape,x:[&[f32];4])->Vec<f32> {
    let (_,_,units,words)=crate::block_ops::checked_lengths(s).unwrap();
    let mut out=vec![0.0f32;units*words];
    for (index,unit) in out.chunks_exact_mut(words).enumerate() {crate::block_attention::unit(s,index,x,&[],unit);}
    out
}
#[test] fn sink_softmax_staged_gives_the_words_of_the_unit() {
    use crate::block_attention::softmax_stage;
    let mut state=0x2545_f491u32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    let mut checked=0;
    for (heads,rows) in [(1u32,1u32),(2,7),(3,32),(2,64),(4,1000),(2,1024),(1,33),(2,1025),(1,2112),(1,3000)] {
        let s=Shape {heads,rows,..shape(Op::SinkSoftmax)};
        assert_eq!(softmax_stage(s),Some(heads as usize));
        assert_eq!(crate::block_ops::geometry(s,0),(heads,crate::block_ops::SUM_THREADS as u32));
        let r=rows as usize;
        for live in [0,1,r/2,r.saturating_sub(1),r] {
            for spread in [0.5f32,8.0,60.0] {
                let scores=(0..heads as usize*r).map(|_|{let v=next();odd(v,((v>>8) as f32/8_388_608.0-1.0)*spread)}).collect::<Vec<_>>();
                let sink=(0..heads).map(|_|{let v=next();odd(v>>3,((v>>8) as f32/8_388_608.0-1.0)*spread)}).collect::<Vec<_>>();
                let step=[live as f32];
                let x=[&scores[..],&sink[..],&step[..],&[][..]];
                assert_eq!(bits(&softmax_staged(s,x)),bits(&by_units(s,x)),"{heads} heads of {rows} rows, {live} live, spread {spread}");
                checked+=1;
            }
        }
        // A step word that is no count: every word NaN in both forms.
        let scores=vec![0.5f32;heads as usize*r];let sink=vec![0.0f32;heads as usize];
        for bad in [0.5f32,(r+1) as f32,f32::NAN] {let x=[&scores[..],&sink[..],&[bad][..],&[][..]];assert_eq!(bits(&softmax_staged(s,x)),bits(&by_units(s,x)));}
    }
    assert!(checked>=100,"{checked}");
    // More than SOFTMAX_GROUP blocks of distances (review REVFAST, LOW 1): a second group of the staged loop, and
    // a last block of one distance.
    for (rows,live) in [(16_385u32,16_385usize),(17_000,16_385),(20_000,19_999)] {
        let s=Shape {heads:1,rows,..shape(Op::SinkSoftmax)};let r=rows as usize;
        assert!(live.div_ceil(BLOCK)>crate::block_attention::SOFTMAX_GROUP);
        let scores=(0..r).map(|_|{let v=next();odd(v,((v>>8) as f32/8_388_608.0-1.0)*8.0)}).collect::<Vec<_>>();
        let x=[&scores[..],&[0.25f32][..],&[live as f32][..],&[][..]];
        assert_eq!(bits(&softmax_staged(s,x)),bits(&by_units(s,x)),"{rows} rows, {live} live");
    }
    // Ties of the top: the sink and scores of +0.0 and -0.0, or of one value, in every order. The top of the unit
    // is the first of them; the staged pick gives the same bits.
    for (sink,first,later) in [(0.0f32,-0.0f32,0.0f32),(-0.0,0.0,-0.0),(f32::NEG_INFINITY,-0.0,0.0),(f32::NEG_INFINITY,0.0,-0.0),(-1.0,2.0,2.0),(f32::NAN,1.0,2.0),(f32::NEG_INFINITY,f32::NAN,f32::NEG_INFINITY)] {
        for rows in [3u32,200,1500] {
            let s=Shape {heads:1,rows,..shape(Op::SinkSoftmax)};let r=rows as usize;
            let mut scores=vec![f32::NEG_INFINITY;r];scores[r/3]=first;scores[r-1]=later;if r>3 {scores[1]=f32::NAN;}
            for live in [r,r-1,r/3+1] {let x=[&scores[..],&[sink][..],&[live as f32][..],&[][..]];assert_eq!(bits(&softmax_staged(s,x)),bits(&by_units(s,x)),"sink {sink} first {first} later {later} rows {rows} live {live}");}
        }
    }
    // A shape the arm refuses is not staged.
    assert_eq!(softmax_stage(Shape {heads:2,rows:64,dim:1,..shape(Op::SinkSoftmax)}),None);
    assert_eq!(softmax_stage(Shape {heads:0,rows:64,..shape(Op::SinkSoftmax)}),None);
    assert_eq!(softmax_stage(Shape {heads:2,rows:64,..shape(Op::WindowScores)}),None);
}
#[test] fn window_values_staged_gives_the_words_of_the_unit() {
    use crate::block_attention::{values_stage,VALUES_CHANNELS};
    let mut state=0x9e37_79b9u32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    let mut checked=0;
    // (heads, dim, taps, rows): rows above BLOCK make two and three blocks of distances.
    for (heads,dim,taps,rows) in [(1u32,16u32,1u32,1u32),(2,32,2,9),(4,64,2,1024),(8,64,4,100),(2,16,1,2100),(3,48,3,1025)] {
        let s=Shape {heads,dim,taps,rows,..shape(Op::WindowValues)};
        let (h,d,r)=(heads as usize,dim as usize,rows as usize);
        assert_eq!(values_stage(s),Some(h*d/VALUES_CHANNELS));
        assert_eq!(crate::block_ops::geometry(s,h*d),((h*d/VALUES_CHANNELS) as u32,crate::block_ops::SUM_THREADS as u32));
        for (live,slot) in [(0,0),(1,r-1),(r/2,r/3),(r,r-1),(r,0),(r.saturating_sub(3).max(1),r/2)] {
            let weights=(0..h*r).map(|_|{let v=next();odd(v>>2,(v>>8) as f32/16_777_216.0)}).collect::<Vec<_>>();
            // Packed BF16 pairs: random words with the low half of each made a BF16 value's high half.
            let plane=(0..r*(h/taps as usize)*d/2).map(|_|{let v=next()&0x7fbf_7fbf;f32::from_bits(v^0x3f00_3f00)}).collect::<Vec<_>>();
            let gate=(0..h*d).map(|_|{let v=next();odd(v>>4,((v>>8) as f32/8_388_608.0-1.0)*4.0)}).collect::<Vec<_>>();
            let step=[live as f32,slot as f32];
            let x=[&weights[..],&plane[..],&gate[..],&step[..]];
            assert_eq!(bits(&values_staged(s,x)),bits(&by_units(s,x)),"{heads} heads of {dim}, taps {taps}, {rows} rows, {live} live at slot {slot}");
            checked+=1;
        }
        let (weights,plane,gate)=(vec![0.25f32;h*r],vec![1.0f32;r*(h/taps as usize)*d/2],vec![1.0f32;h*d]);
        for bad in [[0.5f32,0.0],[(r+1) as f32,0.0],[1.0,r as f32],[f32::NAN,0.0]] {let x=[&weights[..],&plane[..],&gate[..],&bad[..]];assert_eq!(bits(&values_staged(s,x)),bits(&by_units(s,x)));}
    }
    assert!(checked>=30,"{checked}");
    // A dim that is no multiple of VALUES_CHANNELS, or a shape the arm refuses, is not staged.
    assert_eq!(values_stage(Shape {heads:2,dim:8,taps:1,rows:4,..shape(Op::WindowValues)}),None);
    assert_eq!(values_stage(Shape {heads:2,dim:24,taps:1,rows:4,..shape(Op::WindowValues)}),None);
    assert_eq!(values_stage(Shape {heads:2,dim:16,taps:0,rows:4,..shape(Op::WindowValues)}),None);
    assert_eq!(values_stage(Shape {heads:2,dim:16,taps:1,rows:4,bound:1.0,..shape(Op::WindowValues)}),None);
    assert_eq!(values_stage(Shape {heads:2,dim:16,taps:1,rows:4,..shape(Op::SinkSoftmax)}),None);
}
/// The rows entry `resident_launch_block_softmax`: block b runs the staged head b mod h on the inputs of row b / h.
fn softmax_rows_staged(s:Shape,rows:usize,shared:u32,x:[&[f32];4])->Vec<f32> {
    let blocks=crate::block_attention::softmax_rows_stage(s,rows,shared).unwrap();
    let (h,r)=(s.heads as usize,s.rows as usize);
    assert_eq!(blocks,rows*h);
    let (one,_,_,_)=crate::block_ops::checked_lengths(s).unwrap();
    let mut out=vec![f32::from_bits(0x7fc0_dead);rows*h*r];
    for block in 0..blocks {
        let (row,head)=(block/h,block%h);
        let words=softmax_staged(s,crate::block_rows::row_inputs(shared,row,one,x));
        out[(row*h+head)*r..(row*h+head+1)*r].copy_from_slice(&words[head*r..(head+1)*r]);
    }
    out
}
#[test] fn sink_softmax_staged_at_rows_gives_the_words_of_the_rows_unit() {
    let mut state=0x1b87_3593u32;
    let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
    let mut checked=0;
    for (heads,keys,rows) in [(1u32,1u32,1usize),(2,7,3),(3,32,5),(2,100,4),(4,1024,3),(1,2112,2),(2,33,17),(1,16_400,2)] {
        let s=Shape {heads,rows:keys,..shape(Op::SinkSoftmax)};
        let (h,r)=(heads as usize,keys as usize);
        for shared in [0u32,0b0010,0b0100,0b0110] {
            let per=|n:usize,words:usize|if shared>>n&1==1 {words} else {rows*words};
            let scores=(0..rows*h*r).map(|_|{let v=next();odd(v,((v>>8) as f32/8_388_608.0-1.0)*30.0)}).collect::<Vec<_>>();
            let sink=(0..per(1,h)).map(|_|{let v=next();odd(v>>3,((v>>8) as f32/8_388_608.0-1.0)*30.0)}).collect::<Vec<_>>();
            // Live counts of each row (or the one shared), some no count (NaN words).
            // A shared step word is no count in every third shape (all words NaN).
            let bad_shared=shared&0b0100!=0 && keys%3==1;
            let step=(0..per(2,1)).map(|i|{let v=next() as usize;if i%5==3 || bad_shared {0.5} else {(v%(r+1)) as f32}}).collect::<Vec<_>>();
            let x=[&scores[..],&sink[..],&step[..],&[][..]];
            let (_,_,units,words)=crate::block_rows::lengths(s,rows,shared).unwrap();
            let mut want=vec![0.0f32;units*words];
            for (index,unit) in want.chunks_exact_mut(words).enumerate() {crate::block_rows::unit(s,rows,shared,index,x,&[],unit);}
            assert_eq!(bits(&softmax_rows_staged(s,rows,shared,x)),bits(&want),"{heads} heads of {keys} keys, {rows} rows, mask {shared}");
            checked+=1;
        }
    }
    assert_eq!(checked,32);
}
#[test] fn sink_softmax_is_staged_at_rows_only_with_its_scores_per_row() {
    use crate::block_attention::softmax_rows_stage;
    let s=Shape {heads:24,rows:8192,..shape(Op::SinkSoftmax)};
    assert_eq!(softmax_rows_stage(s,1024,0),Some(24*1024));
    assert_eq!(softmax_rows_stage(s,1024,0b0110),Some(24*1024));
    // Shared scores, the list or window bits, an unused input, another arm, no rows, a shape the arm refuses: refused.
    for mask in [0b0001,0b0111,0b1000,crate::block_rows::LIST,crate::block_rows::WINDOW] {assert_eq!(softmax_rows_stage(s,1024,mask),None,"{mask}");}
    assert_eq!(softmax_rows_stage(Shape {op:Op::WindowValues,..s},1024,0),None);
    assert_eq!(softmax_rows_stage(s,0,0),None);
    assert_eq!(softmax_rows_stage(Shape {dim:1,..s},4,0),None);
    // Review REVSOFTMAX, LOW 1: too many rows, no keys or too many, no heads, more output words than the device ABI holds.
    assert_eq!(softmax_rows_stage(s,crate::block_rows::ROWS_MAX+1,0),None);
    assert_eq!(softmax_rows_stage(Shape {rows:0,..s},4,0),None);
    assert_eq!(softmax_rows_stage(Shape {rows:(1<<24)+1,..s},4,0),None);
    assert_eq!(softmax_rows_stage(Shape {heads:0,..s},4,0),None);
    assert_eq!(softmax_rows_stage(Shape {rows:1<<20,..s},crate::block_rows::ROWS_MAX,0),None);
}
