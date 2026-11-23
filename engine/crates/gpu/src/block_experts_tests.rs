//! Host tests of the arms in `block_experts.rs`: small shapes with hand-worked values.
//! Each expected number is worked out in the comment beside it. The shared `tanh` and `sigmoid` are used only where
//! their value is exact: `tanh(0) = 0`, `tanh` of a large number is 1, `tanh` of a tiny number is that number, and
//! the sigmoid of 100 is 1. Each test states the facts it stands on as its first lines.
use crate::{block_experts,causal_ops::{Op,Shape,reference},fixed_math::{rn,sigmoid,tanh}};
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
/// A spread of FP32 values that are not BF16 values, with both signs.
fn spread(count:usize,seed:usize)->Vec<f32> {(0..count).map(|i|(((i*7+seed*13)%41) as f32-20.0)/3.0+0.001).collect()}
/// The reciprocal of a cap as the host makes it (ruling R8): the 64-bit `1 / cap`, rounded one time to 32-bit.
fn recip(cap:f32)->f32 {(1.0/f64::from(cap)) as f32}
/// The weight of `CappedGate`: the reciprocal of the gate cap, then the reciprocal of the up cap.
fn recips(gate_cap:f32,up_cap:f32)->[f32;2] {[recip(gate_cap),recip(up_cap)]}
fn capped(picks:u32,width:u32,gate_cap:f32,up_cap:f32)->Shape {Shape {op:Op::CappedGate,heads:0,dim:width,taps:0,rows:picks,cols:0,epsilon:gate_cap,bound:up_cap}}
fn fold(picks:u32,width:u32)->Shape {Shape {op:Op::RouteFold,heads:0,dim:width,taps:0,rows:picks,cols:0,epsilon:0.0,bound:0.0}}
fn softcap(count:u32,cap:f32)->Shape {Shape {op:Op::Softcap,heads:0,dim:0,taps:0,rows:count,cols:0,epsilon:recip(cap),bound:cap}}
/// 2^-20: a tiny number. 2^24: the FP32 sum `2^24 + 1` is a tie and stays `2^24`.
const TINY:f32=1.0/1_048_576.0;
const BIG:f32=16_777_216.0;
/// 17 * 2^-24 = 2^-20 * 1.0625, a BF16 value. Its product with the reciprocal of 15 is not its quotient by 15.
const SEVENTEEN:f32=17.0/16_777_216.0;

#[test] fn capped_gate_hand_values_and_the_order_of_the_halves() {
    for x in [25.0f32,40.0,250.0] {assert_eq!(tanh(x),1.0);assert_eq!(tanh(-x),-1.0);}
    assert_eq!(tanh(0.0),0.0);assert_eq!(sigmoid(100.0),1.0);assert_eq!(sigmoid(0.0),0.5);
    assert_eq!(recip(4.0),0.25);assert_eq!(tanh(1000.0*recip(25.0)),1.0);
    // 2 picks of width 2, gate cap 4, up cap 25. A pick is its gate half and then its up half.
    let input=[100.0,100.0, 1000.0,-1000.0,   0.0,100.0, 1000.0,0.0];
    let got=reference(capped(2,2,4.0,25.0),[&input,&[],&[],&[]],&recips(4.0,25.0)).unwrap();
    // Pick 0, element 0: gate 100 gives cap 4 * tanh(25) = 4, times sigmoid(100) = 1. Up 1000 gives 25. 4 * 25 = 100.
    // Pick 0, element 1: gate 100 and up -1000: 4 * -25 = -100.
    // Pick 1, element 0: gate 0 gives 4 * tanh(0) = 0, times sigmoid(0) = 1/2, times 25 = 0.
    // Pick 1, element 1: gate 100 and up 0: 4 * 0 = 0.
    // The sigmoid reads the gate itself: with the capped gate 4 the first result is 98.2. With the two halves side by
    // side, or with the caps the other way, the first result is 99.9.
    assert_eq!(got,[100.0,-100.0,0.0,0.0]);
}
#[test] fn capped_gate_takes_each_reciprocal_from_its_weight_word_and_each_cap_from_its_field() {
    // The arm does not test that a reciprocal goes with its cap, so words that do not go together show the places.
    let (small,quarter)=(0.75*TINY,0.375*TINY);
    assert_eq!(tanh(25.0),1.0);assert_eq!(tanh(50.0),1.0);assert_eq!(tanh(500.0),1.0);assert_eq!(sigmoid(100.0),1.0);
    assert_eq!(tanh(small),small);assert_eq!(tanh(quarter),quarter);
    // Caps 3 and 5, weight words 1/4 and 1/2. Gate 100: tanh(100 / 4) * 3 = 3. Up 1.5 * 2^-20: tanh of 0.75 * 2^-20
    // is itself, times 5 = 3.75 * 2^-20. 3 * 3.75 = 11.25. A division by the caps gives 4.5 * 2^-20.
    let got=reference(capped(1,1,3.0,5.0),[&[100.0,1.5*TINY],&[],&[],&[]],&[0.25,0.5]).unwrap();
    assert_eq!(got,[11.25*TINY]);
    // The two weight words the other way: the up half is 0.375 * 2^-20 * 5, and 3 * 1.875 = 5.625.
    let got=reference(capped(1,1,3.0,5.0),[&[100.0,1.5*TINY],&[],&[],&[]],&[0.5,0.25]).unwrap();
    assert_eq!(got,[5.625*TINY]);
    // The gate half reads word 0: caps 2 and 4, weight words 2^-27 and 1/2. Gate 100 gives 100 * 2^-27 =
    // 0.78125 * 2^-20, its tanh is itself, times 2 = 1.5625 * 2^-20. Up 1000 gives tanh(500) * 4 = 4. Result 6.25.
    let low=100.0/134_217_728.0f32;
    assert_eq!(tanh(low),low);assert_eq!(low,0.78125*TINY);
    let got=reference(capped(1,1,2.0,4.0),[&[100.0,1000.0],&[],&[],&[]],&[1.0/134_217_728.0,0.5]).unwrap();
    assert_eq!(got,[6.25*TINY]);
}
#[test] fn capped_gate_multiplies_by_the_reciprocal_and_does_not_divide() {
    // Ruling R8. Up cap 15: r = 8947849 * 2^-27 (bits 0x3d888889), the round of 1/15. Up half x = 17 * 2^-24.
    // x * r = 152113433 * 2^-51, which rounds to q = 9507090 * 2^-47. tanh(q) = q. q * 15 = 142606350 * 2^-47, which
    // rounds to 8912897 * 2^-43 = x + 2^-43. A division gives x / 15 = 9507089 * 2^-47, and that times 15 is x.
    let (x,r)=(SEVENTEEN,recip(15.0));let q=x*r;
    assert_eq!(r.to_bits(),0x3d88_8889);assert_eq!(q.to_bits(),0x3391_1112);assert_eq!(tanh(q),q);assert_eq!(rn(x),x);
    assert_eq!((q*15.0).to_bits(),x.to_bits()+1);assert_eq!(((x/15.0)*15.0).to_bits(),x.to_bits());
    // Gate cap 1.0625, gate 100: the gate half is 1.0625 * 1 * 1.
    assert_eq!(tanh(100.0*recip(1.0625)),1.0);assert_eq!(sigmoid(100.0),1.0);
    let got=reference(capped(1,1,1.0625,15.0),[&[100.0,x],&[],&[],&[]],&recips(1.0625,15.0)).unwrap();
    // 1.0625 * (x + 2^-43) is 2^-20 * (144.5 / 128) and a little more: above the BF16 tie, so it rounds up to
    // 2^-20 * 145 / 128. With the division the product is the tie itself, which goes to the even value 144 / 128.
    assert_eq!(got,[145.0/128.0*TINY]);
    assert_eq!(rn(1.0625*x),144.0/128.0*TINY);
}
#[test] fn capped_gate_rounds_its_halves_as_it_reads_them() {
    // The up half 2^-20 * (1 + 2^-8 + 2^-12) is read as its round u = 2^-20 * (1 + 2^-7). With the up cap 1 its cap
    // is tanh(u) = u. The gate 100 with the gate cap 3 gives 3 * 1 * 1 = 3.
    let (loose,u)=(TINY*(1.0+1.0/256.0+1.0/4096.0),TINY*1.0078125);
    assert_eq!(rn(loose),u);assert_eq!(tanh(u),u);assert_eq!(tanh(100.0*recip(3.0)),1.0);
    let got=reference(capped(1,1,3.0,1.0),[&[100.0,loose],&[],&[],&[]],&recips(3.0,1.0)).unwrap();
    // 3 * u = 2^-19 * (1.5 + 2^-7 + 2^-8): a BF16 tie that goes to the even value 2^-19 * (1.5 + 2^-6).
    assert_eq!(got,[1.515625/524_288.0]);
    // Read as it is, the up half gives 2^-19 * 1.5062 and the round 2^-19 * (1.5 + 2^-7).
    assert_eq!(rn(3.0*loose),1.5078125/524_288.0);
}
#[test] fn capped_gate_rounds_its_result_one_time_at_the_end() {
    // Gate cap 1 + 2^-8, not a BF16 value: the gate 100 gives that cap times 1. The up half 1.5 * 2^-20 with the up
    // cap 1 gives itself.
    let (gate_cap,up)=(1.0+1.0/256.0f32,1.5*TINY);
    assert_eq!(tanh(up),up);assert_eq!(tanh(100.0*recip(gate_cap)),1.0);assert_eq!(rn(gate_cap),1.0);
    let got=reference(capped(1,1,gate_cap,1.0),[&[100.0,up],&[],&[],&[]],&recips(gate_cap,1.0)).unwrap();
    // (1 + 2^-8) * 1.5 * 2^-20 = 2^-20 * (1.5 + 2^-8 + 2^-9), which rounds up to 2^-20 * (1.5 + 2^-7).
    // A round of the first product gives 1 and the result 2^-20 * 1.5.
    assert_eq!(got,[1.5078125*TINY]);
}
#[test] fn capped_gate_equals_the_activation_written_out() {
    let (picks,width)=(3usize,5usize);let input=spread(picks*2*width,1);let weight=recips(4.0,25.0);
    let got=reference(capped(picks as u32,width as u32,4.0,25.0),[&input,&[],&[],&[]],&weight).unwrap();
    let mut expected=Vec::new();
    for pick in 0..picks {for element in 0..width {
        let (gate,up)=(rn(input[pick*2*width+element]),rn(input[pick*2*width+width+element]));
        let (gate_ratio,up_ratio)=(gate*weight[0],up*weight[1]);
        let (gate_capped,up_capped)=(tanh(gate_ratio)*4.0,tanh(up_ratio)*25.0);
        let gated=gate_capped*sigmoid(gate);
        expected.push(rn(gated*up_capped));
    }}
    assert_eq!(bits(&got),bits(&expected));
    assert!(got.iter().all(|&v|rn(v).to_bits()==v.to_bits()));
}
#[test] fn capped_gate_refuses_a_shape_with_no_size_or_no_cap_and_a_weight_that_is_not_two_words() {
    assert_eq!(capped(3,5,4.0,25.0).lengths().unwrap(),([30,0,0,0],2,15,1));
    for bad in [capped(0,5,4.0,25.0),capped(3,0,4.0,25.0),capped(3,5,0.0,25.0),capped(3,5,4.0,0.0),capped(3,5,-4.0,25.0),capped(3,5,4.0,-25.0),
        capped(3,5,f32::NAN,25.0),capped(u32::MAX,u32::MAX,4.0,25.0)] {assert!(bad.lengths().is_none(),"{bad:?}");}
    let weight=recips(4.0,25.0);
    assert!(reference(capped(1,2,4.0,25.0),[&[1.0;4],&[],&[],&[]],&weight).is_ok());
    assert!(reference(capped(1,2,4.0,25.0),[&[1.0;3],&[],&[],&[]],&weight).is_err());
    for wrong in [&[][..],&[0.25],&[0.25,0.04,0.04]] {assert!(reference(capped(1,2,4.0,25.0),[&[1.0;4],&[],&[],&[]],wrong).is_err());}
}

#[test] fn route_fold_hand_values() {
    // 2 picks of width 2. Weights 1/2 and 2. Rows [4, 8] and [1, -1].
    let got=reference(fold(2,2),[&[0.5,2.0],&[4.0,8.0, 1.0,-1.0],&[],&[]],&[]).unwrap();
    // 4/2 + 1 * 2 = 4 and 8/2 - 1 * 2 = 2.
    assert_eq!(got,[4.0,2.0]);
    // 1 pick: the row times its weight.
    assert_eq!(reference(fold(1,3),[&[0.25],&[4.0,-8.0,0.0],&[],&[]],&[]).unwrap(),[1.0,-2.0,0.0]);
}
#[test] fn route_fold_adds_the_picks_in_the_order_given() {
    // Products 2^24, 1, 1, -2^24 in pick order: 2^24 + 1 stays 2^24 two times, then - 2^24 = 0.
    let got=reference(fold(4,1),[&[1.0;4],&[BIG,1.0,1.0,-BIG],&[],&[]],&[]).unwrap();
    assert_eq!(got[0].to_bits(),0);
    // The same picks given in the opposite order: -2^24 + 1 + 1 + 2^24 = 2. The arm does not sort the picks.
    let got=reference(fold(4,1),[&[1.0;4],&[-BIG,1.0,1.0,BIG],&[],&[]],&[]).unwrap();
    assert_eq!(got,[2.0]);
    // The weight of a pick goes with its row: 2 * 3 + 0.5 * 8 = 10, not 0.5 * 3 + 2 * 8.
    assert_eq!(reference(fold(2,1),[&[2.0,0.5],&[3.0,8.0],&[],&[]],&[]).unwrap(),[10.0]);
}
#[test] fn route_fold_rounds_the_rows_as_it_reads_them_and_its_sum_one_time() {
    // The row value 1 + 2^-8 + 2^-12 is read as its round 1 + 2^-7. Times 3 that is 3.0234375, a BF16 tie that goes
    // to the even value 3.03125. Read as it is, the value gives 3.0124 and the round 3.015625.
    let loose=1.0+1.0/256.0+1.0/4096.0f32;
    assert_eq!(rn(loose),1.0078125);assert_eq!(rn(3.0*loose),3.015625);
    assert_eq!(reference(fold(1,1),[&[3.0],&[loose],&[],&[]],&[]).unwrap(),[3.03125]);
    // The weights stay FP32, and the sum is rounded one time. 3 picks with the weight 1 + 2^-8 and the row value 1:
    // the sum is 3 + 3 * 2^-8 = 3.01171875, which rounds up to 3.015625. A round of each weight (1, a tie) or of each
    // product gives 3.
    let weight=1.0+1.0/256.0f32;
    assert_eq!(rn(weight),1.0);
    assert_eq!(reference(fold(3,1),[&[weight;3],&[1.0;3],&[],&[]],&[]).unwrap(),[3.015625]);
}
#[test] fn route_fold_equals_the_sum_written_out() {
    let (picks,width)=(8usize,5usize);let (weights,rows)=(spread(picks,2),spread(picks*width,3));
    let got=reference(fold(picks as u32,width as u32),[&weights,&rows,&[],&[]],&[]).unwrap();
    let expected=(0..width).map(|channel|{let mut sum=0.0f32;for pick in 0..picks {let product=rn(rows[pick*width+channel])*weights[pick];sum+=product;}rn(sum)}).collect::<Vec<_>>();
    assert_eq!(bits(&got),bits(&expected));
    assert_eq!(fold(3,5).lengths().unwrap(),([3,15,0,0],0,5,1));
    for bad in [fold(0,5),fold(3,0),fold(u32::MAX,u32::MAX)] {assert!(bad.lengths().is_none(),"{bad:?}");}
    assert!(reference(fold(2,2),[&[0.5],&[4.0,8.0,1.0,-1.0],&[],&[]],&[]).is_err());
    assert!(reference(fold(2,2),[&[0.5,2.0],&[4.0,8.0,1.0],&[],&[]],&[]).is_err());
}

#[test] fn softcap_hand_values() {
    assert_eq!(tanh(20.0),1.0);assert_eq!(tanh(1000.0*recip(30.0)),1.0);assert_eq!(tanh(600.0*recip(30.0)),1.0);
    // Cap 30. tanh(0) = 0 gives 0. A value far past the cap gives the cap with the sign of the value.
    let got=reference(softcap(5,30.0),[&[0.0,1000.0,-1000.0,600.0,-600.0],&[],&[],&[]],&[]).unwrap();
    assert_eq!(got,[0.0,30.0,-30.0,30.0,-30.0]);assert_eq!(got[0].to_bits(),0);
    // A tiny value comes back as it is: tanh(2^-20 * 1) * 1.
    assert_eq!(tanh(TINY),TINY);assert_eq!(recip(1.0),1.0);
    assert_eq!(reference(softcap(2,1.0),[&[TINY,-TINY],&[],&[],&[]],&[]).unwrap(),[TINY,-TINY]);
}
#[test] fn softcap_takes_the_reciprocal_from_epsilon_and_the_cap_from_bound() {
    // The arm does not test that the two go together. Reciprocal 1/2, cap 3. The value 1.5 * 2^-20 gives
    // tanh(0.75 * 2^-20) * 3 = 2.25 * 2^-20. The value 1000 gives tanh(500) * 3 = 3. A division by the cap gives
    // 1.5 * 2^-20 for the first value.
    let small=0.75*TINY;
    assert_eq!(tanh(small),small);assert_eq!(tanh(500.0),1.0);
    let shape=Shape {op:Op::Softcap,heads:0,dim:0,taps:0,rows:2,cols:0,epsilon:0.5,bound:3.0};
    assert_eq!(reference(shape,[&[1.5*TINY,1000.0],&[],&[],&[]],&[]).unwrap(),[2.25*TINY,3.0]);
}
#[test] fn softcap_multiplies_by_the_reciprocal_and_does_not_divide() {
    // Ruling R8. Cap 15: r = 8947849 * 2^-27 (bits 0x3d888889), the round of 1/15. x = 17 * 2^-24, a BF16 value.
    // x * r = 152113433 * 2^-51, which rounds to q = 9507090 * 2^-47. tanh(q) = q. q * 15 = 142606350 * 2^-47, which
    // rounds to 8912897 * 2^-43 = x + 2^-43: one FP32 step above x.
    let (x,r)=(SEVENTEEN,recip(15.0));let q=x*r;
    assert_eq!(r.to_bits(),0x3d88_8889);assert_eq!(q.to_bits(),0x3391_1112);assert_eq!(tanh(q),q);assert_eq!(rn(x),x);
    let got=reference(softcap(2,15.0),[&[x,-x],&[],&[],&[]],&[]).unwrap();
    assert_eq!(got[0].to_bits(),x.to_bits()+1);assert_eq!(got[1].to_bits(),(-x).to_bits()+1);
    // A division gives x / 15 = 9507089 * 2^-47, and that times 15 = 142606335 * 2^-47 rounds to x itself.
    let divided=x/15.0;
    assert_eq!(divided.to_bits(),0x3391_1111);assert_eq!(tanh(divided),divided);assert_eq!((divided*15.0).to_bits(),x.to_bits());
}
#[test] fn softcap_rounds_as_it_reads_and_keeps_fp32() {
    // 2^-20 * (1 + 2^-10) is not a BF16 value. It is read as 2^-20, and with the cap 1 the result is 2^-20.
    let loose=TINY*(1.0+1.0/1024.0);
    assert_eq!(rn(loose),TINY);assert_eq!(tanh(TINY),TINY);assert_ne!(loose,TINY);
    assert_eq!(reference(softcap(1,1.0),[&[loose],&[],&[],&[]],&[]).unwrap(),[TINY]);
    // The result is not rounded: with the cap 1 + 2^-10 a large value gives that cap, which is not a BF16 value.
    let cap=1.0+1.0/1024.0f32;
    assert_eq!(tanh(1000.0*recip(cap)),1.0);assert_ne!(rn(cap),cap);
    assert_eq!(reference(softcap(1,cap),[&[1000.0],&[],&[],&[]],&[]).unwrap(),[cap]);
}
#[test] fn softcap_equals_the_cap_written_out() {
    let values=spread(23,4).iter().map(|&v|v*9.0).collect::<Vec<_>>();
    let got=reference(softcap(23,30.0),[&values,&[],&[],&[]],&[]).unwrap();
    let expected=values.iter().map(|&v|{let ratio=rn(v)*recip(30.0);tanh(ratio)*30.0}).collect::<Vec<_>>();
    assert_eq!(bits(&got),bits(&expected));
    assert!(got.iter().all(|v|v.abs()<=30.0));assert!(got.iter().any(|&v|rn(v).to_bits()!=v.to_bits()));
    assert_eq!(softcap(7,30.0).lengths().unwrap(),([7,0,0,0],0,7,1));
    for bad in [softcap(0,30.0),softcap(7,0.0),softcap(7,-30.0),softcap(7,f32::INFINITY),Shape {epsilon:0.0,..softcap(7,30.0)},
        Shape {epsilon:-0.5,..softcap(7,30.0)},Shape {epsilon:f32::NAN,..softcap(7,30.0)},Shape {bound:0.0,..softcap(7,30.0)}] {assert!(bad.lengths().is_none(),"{bad:?}");}
    assert!(reference(softcap(2,30.0),[&[1.0],&[],&[],&[]],&[]).is_err());
}
#[test] fn every_unit_of_an_expert_arm_writes_every_word_of_its_slice() {
    for s in [capped(3,5,4.0,25.0),fold(8,5),softcap(23,30.0)] {
        let (lengths,weight_words,units,words)=s.lengths().unwrap();let weight=vec![0.25f32;weight_words];
        let inputs=lengths.map(|count|spread(count,count));let refs=[&inputs[0][..],&inputs[1][..],&inputs[2][..],&inputs[3][..]];
        let mut out=vec![f32::from_bits(0x7fc0_dead);units*words];
        for (index,slice) in out.chunks_exact_mut(words).enumerate() {block_experts::unit(s,index,refs,&weight,slice);}
        assert!(out.iter().all(|v|v.to_bits()!=0x7fc0_dead && v.is_finite()),"{:?}",s.op);
        // A unit of another file's arm writes nothing here.
        let mut other=[f32::from_bits(0x7fc0_dead)];block_experts::unit(Shape {op:Op::PlainNorm,..s},0,refs,&weight,&mut other);
        assert_eq!(other[0].to_bits(),0x7fc0_dead);
    }
    assert!(block_experts::lengths(Shape {op:Op::PlainNorm,..softcap(7,30.0)}).is_none());
}
