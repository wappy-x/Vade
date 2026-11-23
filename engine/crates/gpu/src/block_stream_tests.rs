//! Host tests of the arms in `block_stream.rs`: small shapes with hand-worked values.
//! Each expected number is worked out in the comment beside it. The values are chosen so that every step is exact
//! in FP32 (small whole numbers, halves, powers of two), or so that the one step that rounds is named.
use crate::{block_ops::{LANES,norm_square_sum},block_stream::{self,CONV_ADDEND,CONV_ROUND_SUM,CONV_TOKEN_TAPS,tap_place},causal_ops::{Op,Shape,reference,root,sigmoid,square_sum},fixed_math::{exp,rn,tanh}};
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
fn shape(op:Op,heads:u32,dim:u32,taps:u32,cols:u32,epsilon:f32,bound:f32)->Shape {Shape {op,heads,dim,taps,rows:0,cols,epsilon,bound}}
/// A spread of FP32 values that are not BF16 values, with both signs.
fn spread(count:usize,seed:usize)->Vec<f32> {(0..count).map(|i|(((i*7+seed*13)%41) as f32-20.0)/9.0+0.001).collect()}
fn conv(heads:u32,dim:u32,taps:u32,form:u32,reciprocal:f32)->Shape {shape(Op::ShortConv,heads,dim,taps,form,0.0,reciprocal)}
/// 2^24. The FP32 sum `2^24 + 1` is a tie and goes to the even value `2^24`, so a sum that meets it shows its order.
const BIG:f32=16_777_216.0;

#[test] fn short_conv_at_the_start_of_the_text_reads_zeros_for_the_past() {
    // One head of 2 channels, 4 taps, reciprocal 1/2. The current token is in slot 0. Slots 1 to 3 hold zeros.
    let ring=[2.0,-3.0, 0.0,0.0, 0.0,0.0, 0.0,0.0];
    // Rows by lag: lag 0 first. Channel 0 weights the current token with 1/2, channel 1 with 2.
    let taps=[0.5,1.0,1.0,1.0, 2.0,1.0,1.0,1.0];
    let got=reference(conv(1,2,4,0,0.5),[&ring,&[],&[],&[0.0]],&taps).unwrap();
    // Channel 0: 0.5 * 2 = 1, plus the token 2 = 3, times 1/2 = 1.5. Channel 1: 2 * -3 = -6, plus -3 = -9, times 1/2 = -4.5.
    assert_eq!(got,[1.5,-4.5]);
}
#[test] fn short_conv_walks_the_ring_back_from_the_slot_of_the_current_token() {
    // One channel, 4 taps, reciprocal 1/4. Slots 0 to 3 hold 1, 2, 4, 8. The taps of lags 0 to 3 are 1, 1/2, 1/4, 1/8.
    let (ring,taps)=([1.0,2.0,4.0,8.0],[1.0,0.5,0.25,0.125]);
    let run=|slot:f32|reference(conv(1,1,4,0,0.25),[&ring,&[],&[],&[slot]],&taps).unwrap()[0];
    // Slot 1: lags 3, 2, 1, 0 are slots 2, 3, 0, 1. 4/8 + 8/4 + 1/2 + 2 = 5, plus the token 2 = 7, times 1/4.
    assert_eq!(run(1.0),1.75);
    // Slot 0: slots 1, 2, 3, 0. 2/8 + 4/4 + 8/2 + 1 = 6.25, plus 1 = 7.25, times 1/4.
    assert_eq!(run(0.0),1.8125);
    // Slot 2: slots 3, 0, 1, 2. 8/8 + 1/4 + 2/2 + 4 = 6.25, plus 4 = 10.25, times 1/4.
    assert_eq!(run(2.0),2.5625);
    // Slot 3, a straight plane: slots 0, 1, 2, 3. 1/8 + 2/4 + 4/2 + 8 = 10.625, plus 8 = 18.625, times 1/4.
    assert_eq!(run(3.0),4.65625);
}
#[test] fn short_conv_adds_the_oldest_lag_first_and_the_token_itself_last() {
    // Slot 3 is the current token. Products by lag 3, 2, 1, 0: 4096 * 4096 = 2^24, then 1, 1 and -2^24.
    let (ring,taps)=([4096.0,1.0,1.0,1.0],[-BIG,1.0,1.0,4096.0]);
    let got=reference(conv(1,1,4,0,0.5),[&ring,&[],&[],&[3.0]],&taps).unwrap();
    // Oldest first: 2^24 + 1 stays 2^24 two times, then - 2^24 = 0. Plus the token 1 = 1, times 1/2 = 0.5.
    // Newest first gives 2 + 1 = 3 and 1.5. The token first gives 0.
    assert_eq!(got,[0.5]);
}
#[test] fn short_conv_adds_the_token_made_tap_of_the_head_of_each_channel() {
    // 2 heads of 2 channels, 2 taps, reciprocal 2. Slot 1 is the current token and slot 0 is one token back.
    let ring=[3.0,5.0,1.0,1.0, 2.0,7.0,4.0,8.0];
    let fixed=[0.5,0.25, 1.0,2.0, 0.0,0.0, 0.0,0.0];
    // Token-made taps, head then lag. Head 0 serves channels 0 and 1. Head 1 serves channels 2 and 3.
    let made=[0.5,0.75, -1.0,-1.5];
    let got=reference(conv(2,2,2,CONV_TOKEN_TAPS,2.0),[&ring,&made,&[],&[1.0]],&fixed).unwrap();
    // Channel 0: taps 1 and 1. 1 * 3 + 1 * 2 = 5, plus 2 = 7, times 2 = 14.
    // Channel 1: taps 1.5 (lag 0) and 2.75 (lag 1). 2.75 * 5 + 1.5 * 7 = 24.25, plus 7 = 31.25, times 2 = 62.5.
    // Channel 2: taps -1 and -1.5. -1.5 * 1 + -1 * 4 = -5.5, plus 4 = -1.5, times 2 = -3.
    // Channel 3: -1.5 * 1 + -1 * 8 = -9.5, plus 8 = -1.5, times 2 = -3.
    assert_eq!(got,[14.0,62.5,-3.0,-3.0]);
    // With no token-made taps the fixed taps alone: channel 2 has taps 0, so it gives the token 4 times 2.
    let plain=reference(conv(2,2,2,0,2.0),[&ring,&[],&[],&[1.0]],&fixed).unwrap();
    // Channel 0: 0.25 * 3 + 0.5 * 2 = 1.75, plus 2 = 3.75, times 2. Channel 1: 2 * 5 + 1 * 7 = 17, plus 7 = 24, times 2.
    assert_eq!(plain,[7.5,48.0,8.0,16.0]);
}
#[test] fn short_conv_rounds_the_ring_and_the_token_made_taps_as_it_reads_them() {
    let loose=1.0+1.0/1024.0; // 1 + 2^-10 is not a BF16 value. Its round is 1.
    assert_eq!(rn(loose),1.0);
    // The token-made tap of lag 1 is read as 1: 1 * 1024 + -1024 * 1 = 0, plus the token 1 = 1.
    // Read as it is, the tap gives 1025 - 1024 = 1 and the result 2.
    let got=reference(conv(1,1,2,CONV_TOKEN_TAPS,1.0),[&[1024.0,1.0],&[0.0,loose],&[],&[1.0]],&[-1024.0,0.0]).unwrap();
    assert_eq!(got,[1.0]);
    // The ring value one token back is read as 1: 1024 * 1 + -1024 * 1 = 0, plus the token 1 = 1.
    let got=reference(conv(1,1,2,0,1.0),[&[loose,1.0],&[],&[],&[1.0]],&[-1024.0,1024.0]).unwrap();
    assert_eq!(got,[1.0]);
    // The token's own value is read as 1 + 2^-7 (its round). Times 3 that is 3.0234375, a BF16 tie that goes to the
    // even value 3.03125. Read as it is, the value gives 3.0124 and the round 3.015625.
    let token=1.0+1.0/256.0+1.0/4096.0;
    assert_eq!(rn(token),1.0078125);assert_eq!(rn(token*3.0),3.015625);
    let got=reference(conv(1,1,1,0,3.0),[&[token],&[],&[],&[0.0]],&[0.0]).unwrap();
    assert_eq!(got,[3.03125]);
}
#[test] fn short_conv_adds_the_addend_in_fp32_and_rounds_that_sum_only_when_asked() {
    // One tap of 0: the result is the token 1 times the reciprocal 1/2. The addend 1 + 2^-20 is not a BF16 value.
    let addend=[1.0+1.0/1_048_576.0f32];
    let run=|form:u32|{let given:&[f32]=if form&CONV_ADDEND!=0 {&addend} else {&[]};
        reference(conv(1,1,1,form,0.5),[&[1.0],&[],given,&[0.0]],&[0.0]).unwrap()[0]};
    assert_eq!(run(0),0.5);
    // The addend is not rounded when it is read, and the sum stays FP32: 1.5 + 2^-20.
    assert_eq!(run(CONV_ADDEND).to_bits(),0x3fc0_0008);
    // With the round bit the sum is a BF16 value.
    assert_eq!(run(CONV_ADDEND|CONV_ROUND_SUM),1.5);
}
#[test] fn short_conv_gives_the_same_words_from_any_start_slot_and_equals_the_sum_written_out() {
    // 2 heads of 3 channels, 4 taps, 7 tokens. Ring A puts token i in slot i mod 4. Ring B starts 2 slots later.
    let (heads,dim,taps,tokens)=(2usize,3usize,4usize,7usize);let width=heads*dim;let reciprocal=1.0/3.0f32;
    let fixed=(0..width*taps).map(|i|((i*5)%13) as f32/16.0-0.375).collect::<Vec<_>>();
    let input=|token:usize|(0..width).map(|c|((token*11+c*7)%19) as f32/8.0-1.125).collect::<Vec<f32>>();
    let made=|token:usize|(0..heads*taps).map(|j|((token*3+j*5)%7) as f32/4.0-0.75).collect::<Vec<f32>>();
    let s=conv(heads as u32,dim as u32,taps as u32,CONV_TOKEN_TAPS,reciprocal);
    let (mut a,mut b)=(vec![0.0f32;taps*width],vec![0.0f32;taps*width]);
    for token in 0..tokens {
        let (now,dynamic)=(input(token),made(token));let (slot_a,slot_b)=(token%taps,(token+2)%taps);
        a[slot_a*width..(slot_a+1)*width].copy_from_slice(&now);b[slot_b*width..(slot_b+1)*width].copy_from_slice(&now);
        let got_a=reference(s,[&a,&dynamic,&[],&[slot_a as f32]],&fixed).unwrap();
        let got_b=reference(s,[&b,&dynamic,&[],&[slot_b as f32]],&fixed).unwrap();
        // The sum written out from the plain list of inputs. A token before the text is zero.
        let expected=(0..width).map(|c|{
            let mut sum=0.0f32;
            for lag in [3usize,2,1,0] {
                let past=if token>=lag {input(token-lag)[c]} else {0.0};
                sum+=(fixed[c*taps+lag]+dynamic[(c/dim)*taps+lag])*past;
            }
            rn((sum+now[c])*reciprocal)
        }).collect::<Vec<_>>();
        assert_eq!(bits(&got_a),bits(&expected),"token {token}");assert_eq!(bits(&got_b),bits(&expected),"token {token}, ring B");
    }
}
#[test] fn short_conv_refuses_a_shape_or_a_slot_that_names_no_ring_row() {
    assert_eq!(tap_place(0,4),0);assert_eq!(tap_place(3,4),3);
    assert_eq!(conv(2,3,4,0,0.5).lengths().unwrap(),([24,0,0,1],24,6,1));
    assert_eq!(conv(2,3,4,CONV_TOKEN_TAPS,0.5).lengths().unwrap(),([24,8,0,1],24,6,1));
    assert_eq!(conv(2,3,4,CONV_ADDEND,0.5).lengths().unwrap(),([24,0,6,1],24,6,1));
    assert_eq!(conv(2,3,4,CONV_TOKEN_TAPS|CONV_ADDEND|CONV_ROUND_SUM,0.5).lengths().unwrap(),([24,8,6,1],24,6,1));
    // No size, no reciprocal, a round bit with no addend, a bit that has no meaning.
    for bad in [conv(0,3,4,0,0.5),conv(2,0,4,0,0.5),conv(2,3,0,0,0.5),conv(2,3,4,0,0.0),conv(2,3,4,0,-0.5),conv(2,3,4,0,f32::NAN),
        conv(2,3,4,CONV_ROUND_SUM,0.5),conv(2,3,4,CONV_ROUND_SUM|CONV_TOKEN_TAPS,0.5),conv(2,3,4,8,0.5),conv(u32::MAX,u32::MAX,u32::MAX,0,0.5)] {
        assert!(bad.lengths().is_none(),"{bad:?}");
    }
    let (ring,taps)=([1.0,2.0,4.0,8.0],[1.0,0.5,0.25,0.125]);let s=conv(1,1,4,0,0.25);
    assert!(reference(s,[&ring,&[],&[],&[3.0]],&taps).is_ok());
    // A slot at or past the ring, a slot below zero and a slot that is no whole number read no row and are refused.
    for slot in [4.0f32,5.0,-1.0,1.5,1.0e9] {assert!(reference(s,[&ring,&[],&[],&[slot]],&taps).is_err(),"slot {slot}");}
    // A wrong extent is refused before any unit runs.
    assert!(reference(s,[&ring[..3],&[],&[],&[0.0]],&taps).is_err());
    assert!(reference(s,[&ring,&[],&[],&[]],&taps).is_err());
    assert!(reference(conv(1,1,4,CONV_TOKEN_TAPS,0.25),[&ring,&[],&[],&[0.0]],&taps).is_err());
    assert!(reference(s,[&ring,&[],&[1.0],&[0.0]],&taps).is_err());
}

fn mix(queries:u32,stack:u32,width:u32,scale:f32)->Shape {shape(Op::MixBlend,queries,width,0,stack,0.0,scale)}
#[test] fn mix_blend_of_equal_scores_is_the_mean_of_the_rows() {
    // Scores 1.5 and 1.5: each exponential is exp(0) = 1, the sum is 2, each weight is 1/2.
    let got=reference(mix(1,2,2,0.5),[&[3.0,3.0],&[2.0,4.0, 6.0,-4.0],&[],&[]],&[]).unwrap();
    // 2/2 + 6/2 = 4 and 4/2 - 4/2 = 0.
    assert_eq!(got,[4.0,0.0]);
    // 4 equal scores: each weight is 1/4. 4/4 + 8/4 + 12/4 + 16/4 = 10.
    let got=reference(mix(1,4,1,1.0),[&[-2.0;4],&[4.0,8.0,12.0,16.0],&[],&[]],&[]).unwrap();
    assert_eq!(got,[10.0]);
    // 1 row in the stack: its weight is exp(0) / exp(0) = 1, so the row comes back bit for bit.
    let row=[1.1f32,-2.3,0.007];
    let got=reference(mix(1,1,3,0.25),[&[7.0],&row,&[],&[]],&[]).unwrap();
    assert_eq!(bits(&got),bits(&row));
}
#[test] fn mix_blend_gives_each_query_its_own_weights_and_scales_the_dots_first() {
    // exp stops at -87. That small number vanishes in a sum with 1 and in a sum with 8.
    let small=exp(-200.0);
    assert!(small>0.0 && 1.0+small==1.0 && small*2.0+8.0==8.0);
    // Query 0: equal scores, weights 1/2 and 1/2: 8/2 + 2/2 = 5. Query 1: scores 200 and 0: weights 1 and `small`: 8.
    let got=reference(mix(2,2,1,1.0),[&[0.0,0.0, 200.0,0.0],&[8.0,2.0],&[],&[]],&[]).unwrap();
    assert_eq!(got,[5.0,8.0]);
    // The same dots 0 and 4 with two scales. Scale 2^-30: the scores are 0 and 4 * 2^-30, both exponentials are 1,
    // and the result is the mean 4. Scale 64: the scores are 0 and 256, and the result is the second row, 6.
    let tiny=1.0/1_073_741_824.0f32;
    assert_eq!(exp(-(4.0*tiny)),1.0);
    let run=|scale:f32|reference(mix(1,2,1,scale),[&[0.0,4.0],&[2.0,6.0],&[],&[]],&[]).unwrap()[0];
    assert_eq!(run(tiny),4.0);assert_eq!(run(64.0),6.0);
}
#[test] fn mix_blend_adds_the_rows_in_rising_order_from_zero() {
    // 4 equal scores, weights 1/4. Products: 2^24, 1, 1, -2^24. In rising order 2^24 + 1 stays 2^24 two times,
    // then - 2^24 = 0. The opposite order gives 2, and a sum in pairs gives 1.
    let got=reference(mix(1,4,1,1.0),[&[0.0;4],&[4.0*BIG,4.0,4.0,-4.0*BIG],&[],&[]],&[]).unwrap();
    assert_eq!(got[0].to_bits(),0);
    let got=reference(mix(1,4,1,1.0),[&[0.0;4],&[-4.0*BIG,4.0,4.0,4.0*BIG],&[],&[]],&[]).unwrap();
    assert_eq!(got,[2.0]);
}
#[test] fn mix_blend_equals_the_softmax_written_out_and_keeps_fp32() {
    let (queries,stack,width)=(3usize,5usize,7usize);let scale=(1.0/(width as f64).sqrt()) as f32;
    let (dots,rows)=(spread(queries*stack,1),spread(stack*width,2));
    let got=reference(mix(queries as u32,stack as u32,width as u32,scale),[&dots,&rows,&[],&[]],&[]).unwrap();
    let mut expected=Vec::new();
    for query in 0..queries {
        // The scores, the largest taken out, the exponentials, their sum in rising order, one division a weight.
        let scores=dots[query*stack..(query+1)*stack].iter().map(|&dot|dot*scale).collect::<Vec<_>>();
        let largest=scores.iter().fold(f32::NEG_INFINITY,|a,&b|a.max(b));
        let exponentials=scores.iter().map(|&score|exp(score-largest)).collect::<Vec<_>>();
        let mut total=0.0f32;for &e in &exponentials {total+=e;}
        let weights=exponentials.iter().map(|&e|e/total).collect::<Vec<_>>();
        for channel in 0..width {let mut sum=0.0f32;for (i,&weight) in weights.iter().enumerate() {sum+=weight*rows[i*width+channel];}expected.push(sum);}
    }
    assert_eq!(bits(&got),bits(&expected));
    // The result is not rounded to BF16.
    assert!(got.iter().any(|&v|rn(v).to_bits()!=v.to_bits()));
}
#[test] fn mix_blend_refuses_a_shape_with_no_size_or_no_scale() {
    assert_eq!(mix(3,5,7,0.5).lengths().unwrap(),([15,35,0,0],0,21,1));
    for bad in [mix(0,5,7,0.5),mix(3,0,7,0.5),mix(3,5,0,0.5),mix(3,5,7,0.0),mix(3,5,7,-1.0),mix(3,5,7,f32::INFINITY)] {assert!(bad.lengths().is_none(),"{bad:?}");}
    assert!(reference(mix(1,2,2,0.5),[&[3.0],&[2.0,4.0,6.0,-4.0],&[],&[]],&[]).is_err());
    assert!(reference(mix(1,2,2,0.5),[&[3.0,3.0],&[2.0,4.0,6.0],&[],&[]],&[]).is_err());
}

fn gate(width:u32)->Shape {shape(Op::GateBlend,0,width,0,0,0.0,0.0)}
#[test] fn gate_blend_with_a_zero_gate_returns_the_kept_vector() {
    // tanh(0) = 0: (1 - 0) * kept + 0 * message = kept, bit for bit.
    let kept=[1.1f32,-2.3,0.007];
    let got=reference(gate(3),[&[0.0;6],&kept,&[5.0,-6.0,7.0],&[]],&[]).unwrap();
    assert_eq!(bits(&got),bits(&kept));
}
#[test] fn gate_blend_reads_the_forget_row_then_the_gain_row() {
    assert_eq!(tanh(20.0),1.0);assert_eq!(tanh(-20.0),-1.0);assert_eq!(tanh(0.0),0.0);
    // Channel 0: forget 0, gain 20: (1 - 0) * 3 + 1 * 7 = 10. Channel 1: forget 20, gain 0: (1 - 1) * 5 + 0 * 11 = 0.
    // With the rows in the other order the result is 0 and 16.
    let got=reference(gate(2),[&[0.0,20.0, 20.0,0.0],&[3.0,5.0],&[7.0,11.0],&[]],&[]).unwrap();
    assert_eq!(got,[10.0,0.0]);
    // The ends of tanh. Forget -20: (1 + 1) * 3 = 6. Gain -20: -1 * 7. Sum -1. Forget 20 and gain 20: the message alone.
    let got=reference(gate(2),[&[-20.0,20.0, -20.0,20.0],&[3.0,5.0],&[7.0,11.0],&[]],&[]).unwrap();
    assert_eq!(got,[-1.0,11.0]);
}
#[test] fn gate_blend_is_two_products_and_one_sum_in_fp32() {
    let width=9usize;let (rows,kept,message)=(spread(2*width,3),spread(width,4),spread(width,5));
    let got=reference(gate(width as u32),[&rows,&kept,&message,&[]],&[]).unwrap();
    let expected=(0..width).map(|c|{let keep=(1.0-tanh(rows[c]))*kept[c];let add=tanh(rows[width+c])*message[c];keep+add}).collect::<Vec<_>>();
    assert_eq!(bits(&got),bits(&expected));
    assert!(got.iter().any(|&v|rn(v).to_bits()!=v.to_bits()));
    assert_eq!(gate(9).lengths().unwrap(),([18,9,9,0],0,9,1));
    assert!(gate(0).lengths().is_none());
    assert!(reference(gate(2),[&[0.0;3],&[3.0,5.0],&[7.0,11.0],&[]],&[]).is_err());
}

fn norm(rows:u32,width:u32,epsilon:f32)->Shape {shape(Op::PlainNorm,rows,width,0,0,epsilon,0.0)}
#[test] fn plain_norm_hand_values() {
    // [2, 2, 2, 2]: squares 16, mean 4, plus epsilon 12 = 16, root 4, scale 1/4.
    assert_eq!(reference(norm(1,4,12.0),[&[2.0;4],&[],&[],&[]],&[]).unwrap(),[0.5;4]);
    // Each row has its own scale. Epsilon 3/4. Means 1/4, 3.25 and 15.25 give roots 1, 2 and 4.
    let got=reference(norm(3,2,0.75),[&[0.5,0.5, 2.5,-0.5, -5.5,0.5],&[],&[],&[]],&[]).unwrap();
    assert_eq!(got,[0.5,0.5, 1.25,-0.25, -1.375,0.125]);
    // A zero row stays zero.
    assert_eq!(reference(norm(1,3,1.0e-6),[&[0.0;3],&[],&[],&[]],&[]).unwrap(),[0.0;3]);
}
#[test] fn plain_norm_keeps_fp32() {
    // x = 1 + 2^-10 is not a BF16 value. Its square is 1 + 2^-9 + 2^-20, exact in FP32. Epsilon makes the sum 4.
    let x=1.0+1.0/1024.0f32;let epsilon=4.0-x*x;
    assert_eq!(x*x+epsilon,4.0);
    let got=reference(norm(1,1,epsilon),[&[x],&[],&[],&[]],&[]).unwrap();
    // Root 2, scale 1/2: 0.5 + 2^-11. The BF16 round of that is 0.5.
    assert_eq!(got[0].to_bits(),0x3f00_2000);assert_eq!(rn(got[0]),0.5);
}
#[test] fn plain_norm_sums_the_squares_on_the_32_lane_tree() {
    assert_eq!(LANES,32);
    // Against one chain. Lanes 0 to 7 hold 2^24, 1, 1, 1, 1, 1, 1, 1. The join at distance 4 gives 2^24 (the 1 is
    // lost in the tie), 2, 2, 2. At distance 2: 2^24 + 2 and 4. At distance 1: 2^24 + 6. One chain gives 2^24.
    let mut row=[0.0f32;16];row[0]=4096.0;for value in &mut row[1..8] {*value=1.0;}
    let got=reference(norm(1,16,1.0e-6),[&row,&[],&[],&[]],&[]).unwrap();
    // Mean (2^24 + 6) / 16 = 1048576.375, exact in FP32. The epsilon is below half a step of that number.
    let mean=(BIG+6.0)/16.0;
    assert_eq!(mean+1.0e-6,mean);assert_eq!(f64::from(mean),1_048_576.375);
    let scale=1.0/mean.sqrt();
    assert_eq!(got[0].to_bits(),(4096.0*scale).to_bits());assert_eq!(got[1].to_bits(),scale.to_bits());assert_eq!(got[8],0.0);
    // The root is one step above 1024, so the scale is two steps below 2^-10. One chain gives the scale 2^-10 and 4.
    assert_eq!(got[0].to_bits(),0x407f_fffe);assert_ne!(got[0],4.0);
    // Against the tree of the norm kernel (chunks of 128). 1,024 values: 4096 first, then a 1 at every index that is
    // a multiple of 32 from 128 on (28 values). All are in lane 0, so each 1 meets 2^24 alone and is lost: the sum is
    // 2^24, the mean 2^14, the root 128. The chunk tree keeps the 28 and gives a root of 128.0001.
    let mut row=vec![0.0f32;1024];row[0]=4096.0;for index in (128..1024).step_by(32) {row[index]=1.0;}
    let got=reference(norm(1,1024,1.0e-6),[&row,&[],&[],&[]],&[]).unwrap();
    assert_eq!(got[0],32.0);assert_eq!(got[128],0.0078125);assert_eq!(got[992],0.0078125);assert_eq!(got[1],0.0);
}
#[test] fn plain_norm_then_a_round_equals_the_gated_norm_with_an_open_gate() {
    // `GatedNorm` uses the 32-lane tree of `causal_ops`. With weights of 1 and a gate that is open (its sigmoid is
    // exactly 1) it is the BF16 round of the plain norm. So the tree here is that tree on these rows.
    assert_eq!(sigmoid(100.0),1.0);
    for width in [1usize,2,5,31,32,33,64,100,127,128,129,255,1000,1024] {
        let rows=3usize;let x=spread(rows*width,width);
        let plain=reference(norm(rows as u32,width as u32,1.0e-6),[&x,&[],&[],&[]],&[]).unwrap();
        let gated=reference(Shape {op:Op::GatedNorm,..norm(rows as u32,width as u32,1.0e-6)},[&x,&vec![100.0;rows*width],&[],&[]],&vec![1.0;width]).unwrap();
        assert_eq!(bits(&plain.iter().map(|&v|rn(v)).collect::<Vec<_>>()),bits(&gated),"width {width}");
    }
}
#[test] fn plain_norm_refuses_a_shape_with_no_size_or_no_epsilon() {
    assert_eq!(norm(3,5,1.0e-6).lengths().unwrap(),([15,0,0,0],0,3,5));
    for bad in [norm(0,5,1.0e-6),norm(3,0,1.0e-6),norm(3,5,0.0),norm(3,5,-1.0e-6),norm(3,5,f32::NAN)] {assert!(bad.lengths().is_none(),"{bad:?}");}
    assert!(reference(norm(1,4,12.0),[&[2.0;3],&[],&[],&[]],&[]).is_err());
}
fn rounded(rows:u32,width:u32,weighted:bool,epsilon:f32)->Shape {shape(Op::RoundedNorm,rows,width,u32::from(weighted),0,epsilon,0.0)}
#[test] fn rounded_norm_hand_values_with_and_without_the_weight() {
    // [3, 3]: squares 18, mean 9, plus epsilon 7 = 16, root 4, scale 1/4: 0.75 each. Weights 2 and 1/2.
    assert_eq!(reference(rounded(1,2,true,7.0),[&[3.0,3.0],&[],&[],&[]],&[2.0,0.5]).unwrap(),[1.5,0.375]);
    assert_eq!(reference(rounded(1,2,false,7.0),[&[3.0,3.0],&[],&[],&[]],&[]).unwrap(),[0.75,0.75]);
    // Each row has its own scale, and all rows use the one weight row. Epsilon 15: the row [1, 1] has the mean 1 and
    // the root 4, the row [7, -7] has the mean 49 and the root 8.
    let got=reference(rounded(2,2,true,15.0),[&[1.0,1.0, 7.0,-7.0],&[],&[],&[]],&[2.0,0.5]).unwrap();
    assert_eq!(got,[0.5,0.125, 1.75,-0.4375]);
    // A zero row stays zero.
    assert_eq!(reference(rounded(1,3,false,1.0e-6),[&[0.0;3],&[],&[],&[]],&[]).unwrap(),[0.0;3]);
}
#[test] fn rounded_norm_reads_fp32_and_rounds_one_time_at_the_end() {
    // x = 2 + 2^-7 is not a BF16 value: its round is 2. The row is [x, 0], so the mean of squares is x^2 / 2 =
    // 2 + 2^-6 + 2^-15, exact in FP32. Epsilon makes the sum 4: root 2, scale 1/2.
    let x=2.0+1.0/128.0f32;let epsilon=4.0-x*x/2.0;
    assert_eq!(rn(x),2.0);assert_eq!(x*x/2.0+epsilon,4.0);
    let got=reference(rounded(1,2,true,epsilon),[&[x,0.0],&[],&[],&[]],&[1.5,1.0]).unwrap();
    // x / 2 = 1 + 2^-8. Times 1.5 = 1.5 + 2^-8 + 2^-9, which rounds up to 1.5 + 2^-7.
    assert_eq!(got,[1.5078125,0.0]);
    // A round of the input gives the scale 0.50098 and the result 1.5. A round before the weight gives 1 (a tie
    // that goes to the even value) and then 1.5.
    assert_eq!(rn(rn(x*0.5)*1.5),1.5);
    // With no weight the result is the round of 1 + 2^-8: the tie goes to 1.
    assert_eq!(reference(rounded(1,2,false,epsilon),[&[x,0.0],&[],&[],&[]],&[]).unwrap(),[1.0,0.0]);
}
#[test] fn rounded_norm_sums_the_squares_on_the_tree_of_the_norm_kernel() {
    // 1,024 values: 4096 first, and a 1 at the indexes 128, 160, 192, 224. All five are in lane 0 of the 32-lane
    // tree, so there each 1 meets 2^24 alone and is lost. The norm tree adds chunk 1 (the four 1s) apart: 2^24 + 4.
    let mut row=vec![0.0f32;1024];row[0]=4096.0;for index in [128,160,192,224] {row[index]=1.0;}
    assert_eq!(square_sum(&row),BIG);assert_eq!(norm_square_sum(&row),BIG+4.0);
    // Mean 2^14 + 2^-8; the epsilon 2^-12 is below half a step of it. Root 128 + 2^-16, scale 2^-7 - 2^-30.
    let mean=16384.0+1.0/256.0f32;let scale=1.0/root(mean);
    assert_eq!(mean+1.0/4096.0,mean);assert_eq!(scale.to_bits(),(1.0f32/128.0).to_bits()-2);
    // The weight of index 0 is 1 + 2^-7 + 2^-8. 4096 * scale = 32 - 2^-18, and times the weight that is one FP32
    // step below 32.375, so it rounds down to 32.25. With the 32-lane sum the scale is 2^-7 and the product is
    // 32.375, a tie that goes to the even value 32.5.
    let mut weight=vec![1.0f32;1024];weight[0]=1.0+1.0/128.0+1.0/256.0;
    let got=reference(rounded(1,1024,true,1.0/4096.0),[&row,&[],&[],&[]],&weight).unwrap();
    assert_eq!(got[0],32.25);assert_eq!(rn((4096.0/128.0)*weight[0]),32.5);
    assert_eq!(got[128],0.0078125);assert_eq!(got[224],0.0078125);assert_eq!(got[1],0.0);
    // The plain norm of the same row uses the 32-lane sum: its word 0 is 32.
    assert_eq!(reference(norm(1,1024,1.0/4096.0),[&row,&[],&[],&[]],&[]).unwrap()[0],32.0);
}
#[test] fn rounded_norm_of_a_narrow_row_is_the_round_of_the_plain_norm_and_a_wide_row_follows_its_own_tree() {
    // In one chunk of 128 values the two trees are the same tree.
    for width in [1usize,2,5,31,32,33,64,100,127,128] {
        let rows=3usize;let x=spread(rows*width,width);
        let plain=reference(norm(rows as u32,width as u32,1.0e-6),[&x,&[],&[],&[]],&[]).unwrap();
        let got=reference(rounded(rows as u32,width as u32,false,1.0e-6),[&x,&[],&[],&[]],&[]).unwrap();
        assert_eq!(bits(&got),bits(&plain.iter().map(|&v|rn(v)).collect::<Vec<_>>()),"width {width}");
    }
    // Wider rows: the arm written out with the shared tree, with a weight row.
    for width in [129usize,255,1000,1024,2304] {
        let rows=2usize;let (x,weight)=(spread(rows*width,width),spread(width,3));
        let got=reference(rounded(rows as u32,width as u32,true,1.0e-6),[&x,&[],&[],&[]],&weight).unwrap();
        let mut expected=Vec::new();
        for row in x.chunks_exact(width) {
            let mean=norm_square_sum(row)/width as f32;let scale=1.0/root(mean+1.0e-6);
            for (&value,&w) in row.iter().zip(&weight) {let scaled=value*scale;expected.push(rn(scaled*w));}
        }
        assert_eq!(bits(&got),bits(&expected),"width {width}");
        assert!(got.iter().all(|&v|rn(v).to_bits()==v.to_bits()));
    }
}
#[test] fn rounded_norm_refuses_a_shape_with_no_size_or_no_epsilon_and_a_wrong_weight() {
    assert_eq!(rounded(3,5,true,1.0e-6).lengths().unwrap(),([15,0,0,0],5,3,5));
    assert_eq!(rounded(3,5,false,1.0e-6).lengths().unwrap(),([15,0,0,0],0,3,5));
    for bad in [rounded(0,5,true,1.0e-6),rounded(3,0,true,1.0e-6),rounded(3,5,true,0.0),rounded(3,5,false,-1.0e-6),rounded(3,5,true,f32::NAN),
        Shape {taps:2,..rounded(3,5,true,1.0e-6)}] {assert!(bad.lengths().is_none(),"{bad:?}");}
    assert!(reference(rounded(1,2,true,7.0),[&[3.0],&[],&[],&[]],&[2.0,0.5]).is_err());
    assert!(reference(rounded(1,2,true,7.0),[&[3.0,3.0],&[],&[],&[]],&[]).is_err());
    assert!(reference(rounded(1,2,true,7.0),[&[3.0,3.0],&[],&[],&[]],&[2.0]).is_err());
    assert!(reference(rounded(1,2,false,7.0),[&[3.0,3.0],&[],&[],&[]],&[2.0,0.5]).is_err());
}
#[test] fn every_unit_of_a_stream_arm_writes_every_word_of_its_slice() {
    let shapes=[conv(2,3,4,CONV_TOKEN_TAPS|CONV_ADDEND|CONV_ROUND_SUM,0.5),conv(2,3,4,0,0.5),mix(3,5,7,0.5),gate(9),norm(3,5,1.0e-6),rounded(3,5,true,1.0e-6),rounded(3,5,false,1.0e-6)];
    for s in shapes {
        let (lengths,weights,units,words)=s.lengths().unwrap();
        let mut inputs=lengths.map(|count|spread(count,count));
        if s.op==Op::ShortConv {inputs[3]=vec![2.0];}
        let weight=spread(weights,7);let refs=[&inputs[0][..],&inputs[1][..],&inputs[2][..],&inputs[3][..]];
        let mut out=vec![f32::from_bits(0x7fc0_dead);units*words];
        for (index,slice) in out.chunks_exact_mut(words).enumerate() {block_stream::unit(s,index,refs,&weight,slice);}
        assert!(out.iter().all(|v|v.to_bits()!=0x7fc0_dead && v.is_finite()),"{:?}",s.op);
        // A unit of another file's arm writes nothing here.
        let mut other=[f32::from_bits(0x7fc0_dead)];block_stream::unit(Shape {op:Op::Softcap,..s},0,refs,&weight,&mut other);
        assert_eq!(other[0].to_bits(),0x7fc0_dead);
    }
    assert!(block_stream::lengths(Shape {op:Op::Softcap,..norm(3,5,1.0e-6)}).is_none());
}
fn dots(inputs:u32,rows:u32,width:u32)->Shape {Shape {rows,..shape(Op::RowDots,inputs,0,0,width,0.0,0.0)}}
#[test] fn row_dots_hand_values() {
    // Weight rows (1, 2, 3) and (0.5, 0, -1). Input rows (1, 1, 1) and (2, 0, 4). The result is weight row then input row.
    let got=reference(dots(2,2,3),[&[1.0,1.0,1.0, 2.0,0.0,4.0],&[],&[],&[]],&[1.0,2.0,3.0, 0.5,0.0,-1.0]).unwrap();
    assert_eq!(got,[6.0,14.0, -0.5,-3.0]);
    // One chain in rising order from +0.0: 2^24 + 1 is a tie and stays 2^24, then - 2^24 gives 0. Any other order
    // of the three terms keeps the 1.
    assert_eq!(reference(dots(1,1,3),[&[BIG,1.0,-BIG],&[],&[],&[]],&[1.0,1.0,1.0]).unwrap(),[0.0]);
    // No value is rounded: 1 + 2^-10 stays.
    assert_eq!(reference(dots(1,1,1),[&[1.0+1.0/1024.0],&[],&[],&[]],&[1.0]).unwrap()[0].to_bits(),0x3f80_2000);
}
/// The staged form of the entry: each dot's chain in tiles of `DOT_TILE` words (`dot_steps` carried across tiles)
/// gives the words of the unit, for widths on both sides of a tile and of the 8 steps a turn; the geometry is a block
/// of `SUM_THREADS` a dot up to `DOT_STAGE_DOTS` dots, and the unit's grid above.
#[test] fn row_dots_in_tiles_give_the_words_of_the_unit() {
    use crate::block_ops::{DOT_STAGE_DOTS,DOT_TILE,SUM_THREADS,dot_stage,geometry};
    for (inputs,rows,width) in [(1u32,1u32,1u32),(2,1,7),(3,3,8),(4,1,9),(2,48,2304),(1,5,1000),(3,2,33),(2,2,511),(2,2,512),(2,2,513),(1,1,1537),(5,3,4096)] {
        let x=spread((inputs*width) as usize,(width+inputs) as usize);let weight=spread((rows*width) as usize,rows as usize);
        let s=dots(inputs,rows,width);let want=reference(s,[&x,&[],&[],&[]],&weight).unwrap();
        let (count,k)=dot_stage(s).unwrap();assert_eq!((count,k),((inputs*rows) as usize,width as usize));
        assert_eq!(geometry(s,count),(count as u32,SUM_THREADS as u32));
        let h=inputs as usize;
        for (dot,&word) in want.iter().enumerate() {
            let (row,line)=(dot/h,dot%h);let (v,w)=(&x[line*k..(line+1)*k],&weight[row*k..(row+1)*k]);
            let (mut sum,mut first)=(0.0f32,0);
            while first<k {let n=(k-first).min(DOT_TILE);sum=block_stream::dot_steps(sum,n,|j|v[first+j],|j|w[first+j]);first+=n;}
            assert_eq!(sum.to_bits(),word.to_bits(),"dot {dot} of {inputs}x{rows}x{width}");
        }
    }
    let many=dots(DOT_STAGE_DOTS as u32+1,1,64);assert!(dot_stage(many).is_none());
    assert_eq!(geometry(many,DOT_STAGE_DOTS+1),((DOT_STAGE_DOTS+1).div_ceil(128) as u32,128));
    assert!(dot_stage(Shape {op:Op::MixBlend,..dots(2,2,8)}).is_none());
}
#[test] fn row_dots_give_the_words_of_project_rows_and_of_project() {
    // Widths on both sides of the 8 steps a turn, 1 to 4 input rows (the most of `ProjectRows`), 1 to 48 weight rows.
    for (inputs,rows,width) in [(1u32,1u32,1u32),(2,1,7),(3,3,8),(4,1,9),(4,3,2304),(2,48,2304),(1,5,1000),(3,2,33)] {
        let x=spread((inputs*width) as usize,(width+inputs) as usize);let weight=spread((rows*width) as usize,rows as usize);
        let got=reference(dots(inputs,rows,width),[&x,&[],&[],&[]],&weight).unwrap();
        let want=reference(Shape {op:Op::ProjectRows,..dots(inputs,rows,width)},[&x,&[],&[],&[]],&weight).unwrap();
        assert_eq!(bits(&got),bits(&want),"{inputs} input rows, {rows} weight rows of {width}");
    }
    // More input rows than `ProjectRows` takes: each input row alone on `Project`, the one-row chain.
    for (inputs,rows,width) in [(5usize,1usize,2304usize),(7,3,100)] {
        let x=spread(inputs*width,width);let weight=spread(rows*width,rows);
        let got=reference(dots(inputs as u32,rows as u32,width as u32),[&x,&[],&[],&[]],&weight).unwrap();
        for line in 0..inputs {
            let one=reference(Shape {op:Op::Project,heads:0,dim:0,taps:0,rows:rows as u32,cols:width as u32,epsilon:0.0,bound:0.0},[&x[line*width..(line+1)*width],&[],&[],&[]],&weight).unwrap();
            for row in 0..rows {assert_eq!(got[row*inputs+line].to_bits(),one[row].to_bits(),"input row {line}, weight row {row}");}
        }
    }
    assert_eq!(dots(4,3,2304).lengths().unwrap(),([9216,0,0,0],6912,12,1));
    for bad in [dots(0,3,5),dots(4,0,5),dots(4,3,0),Shape {dim:1,..dots(4,3,5)},Shape {taps:1,..dots(4,3,5)},Shape {epsilon:1.0,..dots(4,3,5)},Shape {bound:1.0,..dots(4,3,5)}] {assert!(bad.lengths().is_none(),"{bad:?}");}
}
