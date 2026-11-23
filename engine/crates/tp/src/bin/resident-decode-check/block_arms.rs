//! Device sweep of the block arms (`gpu/src/block_ops.rs`), level 1 of the check program of the loop family.
//! For every arm that is written: every output word of the general kernel on the device against
//! `causal_ops::reference` on the host. Both run the same shared unit code, so each word must be equal. An arm with
//! no body has no lengths; the sweep names it and goes on.
//!
//! The sweep also shows that its data can tell a fused step. An arm that has a product and then a sum has a host
//! twin here, with each such pair of steps in the function `step`. With the steps apart the twin must give the words
//! of the reference. With a fused step it must give another word in a case of that arm, or the arm does not pass.
//! Thus equal device words show that the GPU build kept the two steps apart. `CappedGate`, `Softcap` and
//! `SinkSoftmax` have no sum after a product, and the shared `tanh` and `sigmoid` take their input apart by its
//! bits, so these arms need no twin.
//!
//! Shapes that the host must refuse are sent too: an operand one word short, a shape with no lengths, and step
//! words that name no ring slot or no live count. The device must refuse the first two before a launch. A wrong
//! step word is data on the device: there every output word must be a NaN.
//!
//! Two more facts are shown on the device: `RouteFold` adds in the order of its picks, and the attention arms give
//! the same words when the same keys lie at another start of the ring.
//!
//! The output is not filled with a mark first. The backend fills the output of these arms with zeros before the
//! launch, because `causal_ops::writes_every_word` does not name them, so a mark does not reach the kernel. The host
//! tests of each arm file show that a unit writes every word of its slice.
//!
//! Sizes here are sweep sizes: small and odd ones, and large ones. Each launch runs two times. The mode
//! `block-arms-host` runs the host half alone, with no device.
use std::{fmt::Write as _,fs,io::Write as _,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{block_attention::{BLOCK,CLIP_EPSILON},block_ops::{join,LANES},block_stream::{CONV_ADDEND,CONV_ROUND_SUM,CONV_TOKEN_TAPS,tap_place},causal_ops::{reference,Op,Shape},fixed_math::{exp,rn,sigmoid,tanh}};
/// Every block arm, in the order of its number, with a small shape of its contract. The shape asks if the arm has
/// lengths, that is, if it is written. It is also the shape of the refused operands.
const ARMS:[(&str,Shape);14]=[
    ("HeadClip",Shape {op:Op::HeadClip,heads:2,dim:4,taps:1,rows:0,cols:1,epsilon:0.0,bound:1.0}),
    ("RelBias",Shape {op:Op::RelBias,heads:2,dim:4,taps:0,rows:0,cols:8,epsilon:0.0,bound:0.0}),
    ("ShortConv",Shape {op:Op::ShortConv,heads:2,dim:3,taps:4,rows:0,cols:3,epsilon:0.0,bound:0.5}),
    ("WindowScores",Shape {op:Op::WindowScores,heads:2,dim:4,taps:1,rows:4,cols:4,epsilon:0.0,bound:0.5}),
    ("SinkSoftmax",Shape {op:Op::SinkSoftmax,heads:2,dim:0,taps:0,rows:4,cols:0,epsilon:0.0,bound:0.0}),
    ("WindowValues",Shape {op:Op::WindowValues,heads:2,dim:4,taps:1,rows:4,cols:0,epsilon:0.0,bound:0.0}),
    ("MixBlend",Shape {op:Op::MixBlend,heads:1,dim:2,taps:0,rows:0,cols:2,epsilon:0.0,bound:0.5}),
    ("GateBlend",Shape {op:Op::GateBlend,heads:0,dim:2,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}),
    ("CappedGate",Shape {op:Op::CappedGate,heads:0,dim:2,taps:0,rows:1,cols:0,epsilon:4.0,bound:25.0}),
    ("RouteFold",Shape {op:Op::RouteFold,heads:0,dim:2,taps:0,rows:2,cols:0,epsilon:0.0,bound:0.0}),
    ("Softcap",Shape {op:Op::Softcap,heads:0,dim:0,taps:0,rows:2,cols:0,epsilon:0.03125,bound:32.0}),
    ("PlainNorm",Shape {op:Op::PlainNorm,heads:1,dim:2,taps:0,rows:0,cols:0,epsilon:1.0e-6,bound:0.0}),
    ("RoundedNorm",Shape {op:Op::RoundedNorm,heads:1,dim:2,taps:1,rows:0,cols:0,epsilon:1.0e-6,bound:0.0}),
    ("RowDots",Shape {op:Op::RowDots,heads:2,dim:0,taps:0,rows:3,cols:5,epsilon:0.0,bound:0.0}),
];
/// Signed zeros, the smallest subnormals, the smallest normal number, BF16 ties and their neighbors, and large values.
const SPECIAL:[u32;16]=[0,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_8000,0x3f80_7fff,0x3f81_8000,0xbf80_8000,0x3f7f_ffff,0x4000_0001,0xc0ff_ffff,0x477f_e000,0xc77f_e000,0x3380_0000,0xb380_0000];
/// Gate and cap words: zero, small, one, far past one, and the negatives. A cap case multiplies them by its cap.
const GRID:[f32;8]=[0.0,1.0e-4,1.0,20.0,-1.0e-4,-1.0,-20.0,-0.0];
/// `count` finite words. The plain words lie in -4 to 4 times `span` and use all 24 bits of an FP32 value, so a
/// product of two of them is not exact as a rule. With `special`, two words of three come from the list above.
/// A span of zero gives +0.0 for every word.
fn fill(count:usize,seed:usize,special:bool,span:f32)->Vec<f32> {
    if span==0.0 {return vec![0.0;count];}
    (0..count).map(|i|{let at=i+seed*7919;
        if special && at%3!=1 {f32::from_bits(SPECIAL[(at/3)%SPECIAL.len()])} else {((at.wrapping_mul(2_654_435_761)%16_777_213) as f32/2_097_152.0-4.0)*span}
    }).collect()
}
/// `count` packed words: two BF16 values a word, the first in the low half, as `PackBf16` writes them.
fn pack(count:usize,seed:usize,special:bool,span:f32)->Vec<f32> {
    fill(2*count,seed,special,span).as_chunks::<2>().0.iter().map(|pair|f32::from_bits((rn(pair[0]).to_bits()>>16)|(rn(pair[1]).to_bits()&0xffff_0000))).collect()
}
/// A change that a case makes to its filled inputs.
#[derive(Clone,Copy)]
enum Tweak {
    Plain,
    /// `MixBlend`: the first two dots of each query are equal.
    EqualDots,
    /// `MixBlend`: one dot of each query is far above the others.
    HighDot,
    /// `GateBlend`, `CappedGate`, `Softcap`: the words of `GRID`, in every pair where the arm has two halves.
    Grid,
    /// `RouteFold`: the weight of the middle pick is zero.
    ZeroWeight,
    /// `RouteFold`: a sum that lies on a tie of two FP32 values only when the product and the sum are two steps.
    FoldTie,
    /// The norms and `HeadClip`: the first row is zeros.
    ZeroRow,
    /// `HeadClip`: the length of the first head is the threshold.
    ClipAt,
    /// `SinkSoftmax`: the sink is far above or far below the scores.
    Sink(f32),
}
/// One launch shape of one arm. `spans` scales the plain words of each input and `weight` those of the weight.
/// `step` holds the step words of the arm: the ring slot of `ShortConv`, the live count of `SinkSoftmax`, the live
/// count and the slot of `WindowScores` and `WindowValues`.
struct Case {name:&'static str,label:String,shape:Shape,spans:[f32;4],weight:f32,step:Vec<f32>,tweak:Tweak}
fn case(name:&'static str,label:String,shape:Shape)->Case {Case {name,label,shape,spans:[1.0;4],weight:1.0,step:Vec::new(),tweak:Tweak::Plain}}
/// Step words that an arm takes, for a shape that only shows the refused operands.
fn plain_step(op:Op)->Vec<f32> {match op {Op::ShortConv=>vec![1.0],Op::SinkSoftmax=>vec![2.0],Op::WindowScores|Op::WindowValues=>vec![2.0,1.0],_=>Vec::new()}}
/// The live counts 1, 2, rows - 1 and rows, each with the slot 0, the last slot, and the slot of a straight plane.
fn live_steps(rows:u32)->Vec<(u32,u32)> {
    let mut all=Vec::new();
    for n in [1,2,rows-1,rows] {for slot in [0,rows-1,n.max(1)-1] {if n>0 && !all.contains(&(n,slot)) {all.push((n,slot));}}}
    all
}
fn empty(op:Op)->Shape {Shape {op,heads:0,dim:0,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0}}
/// The reciprocal of a number as the host makes it (ruling R8): the 64-bit `1 / x`, rounded one time to 32-bit.
fn recip(x:f32)->f32 {(1.0/f64::from(x)) as f32}
/// The bits of a weight `w` for `Tweak::FoldTie`: the FP32 product `1.0078125 * w` is 2^-8 + 2^-24, and the exact
/// product is a little above that number.
const FOLD_TIE_WEIGHT:u32=0x3b7e_04f7;
fn cases()->Vec<Case> {
    let mut all=Vec::new();
    // HeadClip: with and without the scale and the factor; heads far below, near and far above the threshold; a
    // head of zeros; a head at the threshold.
    for (heads,dim) in [(1u32,2u32),(3,16),(8,128),(24,128),(24,16)] {for (scale,factor) in [(0u32,0u32),(1,0),(0,1),(1,1)] {
        let shape=|bound:f32|Shape {heads,dim,taps:scale,cols:factor,bound,..empty(Op::HeadClip)};
        let near=2.3*(dim as f32).sqrt();
        for bound in [1.0e6f32,near,1.0e-3] {all.push(case("HeadClip",format!("heads={heads} dim={dim} scale={scale} factor={factor} threshold={bound}"),shape(bound)));}
        all.push(Case {tweak:Tweak::ZeroRow,..case("HeadClip",format!("heads={heads} dim={dim} scale={scale} factor={factor} threshold={near} a head of zeros"),shape(near))});
        all.push(Case {tweak:Tweak::ClipAt,..case("HeadClip",format!("heads={heads} dim={dim} scale={scale} factor={factor} threshold=2.5 a head at the threshold"),shape(2.5))});
    }}
    // A rounded result shows a fused step in about one word of 100,000, so the clip and the relative term get one
    // size with many heads for the fused-step check.
    all.push(case("HeadClip","heads=65536 dim=128 scale=1 factor=0 threshold=26 many heads".into(),Shape {heads:65_536,dim:128,taps:1,bound:26.0,..empty(Op::HeadClip)}));
    all.push(case("RelBias","heads=512 dim=16 extent=1024 many heads".into(),Shape {heads:512,dim:16,cols:1024,..empty(Op::RelBias)}));
    // RelBias: an extent of 1, 4 and 1024, a state width of 2 and 16.
    for heads in [1u32,3,24] {for dim in [2u32,16] {for extent in [1u32,4,1024] {
        all.push(case("RelBias",format!("heads={heads} dim={dim} extent={extent}"),Shape {heads,dim,cols:extent,..empty(Op::RelBias)}));
    }}}
    // WindowScores and WindowValues: a group of 1 and of 3; 4 to 4096 rows; the live counts and slots of `live_steps`;
    // a bias extent below, at and above the rows; a gate of zeros. SinkSoftmax: the same rows and live counts, and a
    // sink far above and far below the scores.
    for rows in [4u32,256,1024,4096] {
        for (heads,dim,group) in [(1u32,2u32,1u32),(3,16,1),(3,16,3),(6,16,3),(8,128,1),(24,128,3)] {
            if rows==4096 && dim==128 && group==1 {continue;}
            for (turn,(n,slot)) in live_steps(rows).into_iter().enumerate() {
                let extent=[1,rows/2,rows,2*rows][turn%4];let scale=(1.0/f64::from(dim).sqrt()) as f32;
                all.push(Case {step:vec![n as f32,slot as f32],..case("WindowScores",format!("heads={heads} dim={dim} group={group} rows={rows} live={n} slot={slot} bias_extent={extent}"),
                    Shape {heads,dim,taps:group,rows,cols:extent,bound:scale,..empty(Op::WindowScores)})});
                let zero_gate=turn%4==3;
                all.push(Case {step:vec![n as f32,slot as f32],spans:[0.25,1.0,if zero_gate {0.0} else {1.0},1.0],
                    ..case("WindowValues",format!("heads={heads} dim={dim} group={group} rows={rows} live={n} slot={slot} zero_gate={}",u8::from(zero_gate)),Shape {heads,dim,taps:group,rows,..empty(Op::WindowValues)})});
            }
        }
        for heads in [1u32,3,24] {for n in [1,2,rows-1,rows] {for (tweak,text,span) in [(Tweak::Plain,"plain",1.0f32),(Tweak::Plain,"wide scores",16.0),(Tweak::Sink(1.0e4),"far above",1.0),(Tweak::Sink(-1.0e4),"far below",1.0)] {
            all.push(Case {step:vec![n as f32],spans:[span,1.0,1.0,1.0],tweak,..case("SinkSoftmax",format!("heads={heads} rows={rows} live={n} sink={text}"),Shape {heads,rows,..empty(Op::SinkSoftmax)})});
        }}}
    }
    // ShortConv: the 6 forms that have lengths, 1 to 5 taps, every slot of the ring, 4 reciprocals, and a ring of zeros.
    let reciprocals=[(1.0/5.0f64.sqrt()) as f32,recip(3.0),0.5,2.0];
    for (heads,dim,taps) in [(1u32,1u32,1u32),(2,3,4),(3,5,2),(7,33,3),(1,1,5),(1,8,1),(1,8,2),(1,8,4),(8,128,1),(8,128,2),(8,128,4),(18,128,4),(1,2304,1),(1,2304,2),(1,2304,4)] {
        for form in [0,CONV_TOKEN_TAPS,CONV_ADDEND,CONV_TOKEN_TAPS|CONV_ADDEND,CONV_ADDEND|CONV_ROUND_SUM,CONV_TOKEN_TAPS|CONV_ADDEND|CONV_ROUND_SUM] {
            for slot in 0..=taps {
                // The last turn is the ring of zeros of a chat start, at slot 0.
                let (zeros,slot)=(slot==taps,slot%taps);let bound=reciprocals[((heads+form+slot)%4) as usize];
                all.push(Case {spans:[if zeros {0.0} else {1.0},0.25,1.0,1.0],weight:0.25,step:vec![slot as f32],
                    ..case("ShortConv",format!("heads={heads} dim={dim} taps={taps} form={form} slot={slot} reciprocal={bound} zero_ring={}",u8::from(zeros)),Shape {heads,dim,taps,cols:form,bound,..empty(Op::ShortConv)})});
            }
        }
    }
    // MixBlend: 1, 3 and 48 queries, a stack of 1 to 9 vectors. Wide scores reach the lower stop of the exponential.
    for (queries,stack,width) in [(1u32,1u32,1u32),(1,2,7),(1,3,2304),(1,4,33),(1,5,2304),(3,2,33),(3,3,5),(3,4,2304),(3,5,33),(48,2,2304),(48,3,33),(48,4,7),(48,5,2304),(2,9,129)] {
        let shape=|bound:f32|Shape {heads:queries,dim:width,cols:stack,bound,..empty(Op::MixBlend)};
        for (bound,span) in [((1.0/f64::from(width).sqrt()) as f32,1.0f32),(1.0,16.0),(0.03125,1.0),(1.0,64.0)] {
            all.push(Case {spans:[span,1.0,1.0,1.0],..case("MixBlend",format!("queries={queries} stack={stack} width={width} scale={bound} span={span}"),shape(bound))});
        }
        if stack>1 {for (tweak,text) in [(Tweak::EqualDots,"two equal dots"),(Tweak::HighDot,"one dot far above")] {
            all.push(Case {tweak,..case("MixBlend",format!("queries={queries} stack={stack} width={width} scale=1 {text}"),shape(1.0))});
        }}
    }
    // GateBlend: small, middle and large gate words, so both branches of the tangent run, and the grid of gate words.
    for width in [1u32,2,33,128,2304,4097] {for span in [1.0f32,8.0,0.015625] {
        all.push(Case {spans:[span,1.0,1.0,1.0],..case("GateBlend",format!("width={width} span={span}"),Shape {dim:width,..empty(Op::GateBlend)})});
    }}
    for width in [64u32,2304] {all.push(Case {tweak:Tweak::Grid,..case("GateBlend",format!("width={width} grid of gate words"),Shape {dim:width,..empty(Op::GateBlend)})});}
    // CappedGate: 1 to 8 picks, 3 pairs of caps, and the grid of gate and up words. `inputs` makes its weight.
    for (gate_cap,up_cap) in [(4.0f32,25.0f32),(1.0,1.0),(0.5,30.0)] {
        let shape=|picks:u32,width:u32|Shape {rows:picks,dim:width,epsilon:gate_cap,bound:up_cap,..empty(Op::CappedGate)};
        for (picks,width) in [(1u32,1u32),(3,5),(8,33),(2,1153),(8,1152)] {for span in [1.0f32,16.0] {
            all.push(Case {spans:[span,1.0,1.0,1.0],..case("CappedGate",format!("picks={picks} width={width} gate_cap={gate_cap} up_cap={up_cap} span={span}"),shape(picks,width))});
        }}
        for (picks,width) in [(1u32,64u32),(8,64),(8,1152)] {
            all.push(Case {tweak:Tweak::Grid,..case("CappedGate",format!("picks={picks} width={width} gate_cap={gate_cap} up_cap={up_cap} grid of gate and up words"),shape(picks,width))});
        }
    }
    // RouteFold: 1 to 32 picks, a weight of zero, and the tie that tells a fused step.
    for (picks,width) in [(1u32,1u32),(3,5),(8,33),(5,384),(32,128),(8,2304),(1,2304)] {
        let shape=Shape {rows:picks,dim:width,..empty(Op::RouteFold)};
        for span in [1.0f32,16.0] {all.push(Case {spans:[0.5,span,1.0,1.0],..case("RouteFold",format!("picks={picks} width={width} span={span}"),shape)});}
        all.push(Case {spans:[0.5,1.0,1.0,1.0],tweak:Tweak::ZeroWeight,..case("RouteFold",format!("picks={picks} width={width} a weight of zero"),shape)});
    }
    for (picks,width) in [(2u32,1u32),(2,33),(8,2304)] {
        all.push(Case {spans:[0.0;4],tweak:Tweak::FoldTie,..case("RouteFold",format!("picks={picks} width={width} a sum on a tie"),Shape {rows:picks,dim:width,..empty(Op::RouteFold)})});
    }
    // Softcap: 3 caps, values below, near and far above the cap, and the grid of cap words.
    for count in [1u32,7,129,4096,200_058] {for (cap,span) in [(30.0f32,1.0f32),(30.0,16.0),(1.0,0.015625),(4.0,1.0),(1.0,64.0)] {
        all.push(Case {spans:[span,1.0,1.0,1.0],..case("Softcap",format!("values={count} cap={cap} span={span}"),Shape {rows:count,epsilon:recip(cap),bound:cap,..empty(Op::Softcap)})});
    }}
    for count in [64u32,200_058] {for cap in [30.0f32,4.0,1.0] {
        all.push(Case {tweak:Tweak::Grid,..case("Softcap",format!("values={count} cap={cap} grid of cap words"),Shape {rows:count,epsilon:recip(cap),bound:cap,..empty(Op::Softcap)})});
    }}
    // PlainNorm and RoundedNorm: 1 to 5 rows, widths on both sides of the 32 lanes and of the chunk of 128 values,
    // wide rows, and a row of zeros. RoundedNorm with and without its weight row.
    // The size with 65,536 rows is there for the fused-step check: a rounded result shows a fused step in about one
    // word of 100,000.
    let mut sizes=vec![(1u32,1u32),(3,5),(1,64),(2,127),(2,128),(2,129),(2,2305),(1,4096),(1,4129),(48,2304),(65_536,33)];
    for rows in 1..=5 {for width in [2u32,31,32,33,2304] {sizes.push((rows,width));}}
    for (name,op,weights) in [("PlainNorm",Op::PlainNorm,&[0u32][..]),("RoundedNorm",Op::RoundedNorm,&[0u32,1][..])] {for &taps in weights {
        for &(rows,width) in &sizes {
            let shape=|epsilon:f32|Shape {heads:rows,dim:width,taps,epsilon,..empty(op)};
            for (epsilon,span) in [(1.0e-6f32,1.0f32),(1.0e-5,64.0),(1.0e-6,0.015625)] {
                all.push(Case {spans:[span,1.0,1.0,1.0],..case(name,format!("rows={rows} width={width} weight={taps} epsilon={epsilon} span={span}"),shape(epsilon))});
            }
            if width>=31 && rows%2==1 {all.push(Case {tweak:Tweak::ZeroRow,..case(name,format!("rows={rows} width={width} weight={taps} a row of zeros"),shape(1.0e-6))});}
        }
    }}
    // RowDots: 1 to 48 weight rows (the queries of the mixer), 1 to 9 input rows (its stack), widths on both sides
    // of the 8 steps of one loop turn, and wide rows. One size with many dots is there for the fused-step check.
    for (queries,stack,width) in [(1u32,1u32,1u32),(1,2,7),(2,3,8),(3,2,9),(1,5,33),(1,2,2304),(1,3,2304),(1,4,2304),(1,5,2304),(3,2,2304),(3,4,2304),(48,2,2304),(2,9,129),(48,5,2304)] {
        for span in [1.0f32,16.0,0.015625] {
            all.push(Case {spans:[span,1.0,1.0,1.0],..case("RowDots",format!("queries={queries} stack={stack} width={width} span={span}"),Shape {heads:stack,rows:queries,cols:width,..empty(Op::RowDots)})});
        }
    }
    all.push(case("RowDots","queries=512 stack=64 width=33 many dots".into(),Shape {heads:64,rows:512,cols:33,..empty(Op::RowDots)}));
    all
}
/// The inputs and the weight of a case. `special` puts the special words in.
fn inputs(case:&Case,special:bool)->Result<([Vec<f32>;4],Vec<f32>)> {
    let s=case.shape;let (lengths,weights,_,_)=s.lengths().ok_or_else(||format!("{} {}: the shape has no lengths",case.name,case.label))?;
    let (h,d,k,r)=(s.heads as usize,s.dim as usize,s.cols as usize,s.rows as usize);
    let mut x:[Vec<f32>;4]=core::array::from_fn(|i|fill(lengths[i],i+1,special,case.spans[i]));
    if !case.step.is_empty() {x[if matches!(s.op,Op::SinkSoftmax) {2} else {3}]=case.step.clone();}
    if matches!(s.op,Op::WindowScores|Op::WindowValues) {x[1]=pack(lengths[1],2,special,case.spans[1]);}
    match (case.tweak,s.op) {
        (Tweak::EqualDots,_)=>for query in 0..h {x[0][query*k+1]=x[0][query*k];},
        (Tweak::HighDot,_)=>for query in 0..h {x[0][query*k+query%k]=1.0e4;},
        (Tweak::Grid,Op::GateBlend)=>for i in 0..d {x[0][i]=GRID[i%8];x[0][d+i]=GRID[(i/8)%8];},
        (Tweak::Grid,Op::CappedGate)=>for pick in 0..r {for e in 0..d {x[0][pick*2*d+e]=GRID[(e+pick)%8]*s.epsilon;x[0][pick*2*d+d+e]=GRID[(e/8+pick)%8]*s.bound;}},
        (Tweak::Grid,_)=>for (i,value) in x[0].iter_mut().enumerate() {*value=GRID[i%8]*s.bound;},
        (Tweak::ZeroWeight,_)=>x[0][r/2]=0.0,
        // Pick 0 gives the product 1. The last pick gives a product a little above 2^-8 + 2^-24, which rounds down to
        // that number. So 1 plus the rounded product is the middle of 1 + 2^-8 and the next FP32 value, and goes to
        // the even one, 1 + 2^-8: a BF16 tie that rounds to 1. A fused step adds the product as it is and gets the
        // next FP32 value, which rounds to 1 + 2^-7. The picks between have the weight zero.
        (Tweak::FoldTie,_)=>{
            x[0][0]=1.0;x[0][r-1]=f32::from_bits(FOLD_TIE_WEIGHT);
            x[1][..d].fill(1.0);x[1][(r-1)*d..].fill(1.0078125);
        },
        (Tweak::ZeroRow,_)=>x[0][..d].fill(0.0),
        (Tweak::ClipAt,_)=>for (j,value) in x[0][..d].iter_mut().enumerate() {*value=if j==0 {s.bound} else {0.0};},
        (Tweak::Sink(sink),_)=>x[1].fill(sink),
        (Tweak::Plain,_)=>{},
    }
    let weight=if matches!(s.op,Op::CappedGate) {vec![recip(s.epsilon),recip(s.bound)]} else {fill(weights,5,special,case.weight)};
    Ok((x,weight))
}
/// One product and then one sum. `fused` makes them one fused step, which no arm may make.
fn step(fused:bool,a:f32,b:f32,sum:f32)->f32 {if fused {a.mul_add(b,sum)} else {sum+a*b}}
/// The sum of squares on 32 lanes: lane = index mod 32, each lane in rising index from +0.0, then the join.
fn lane_squares(x:&[f32],fused:bool)->f32 {
    let mut lanes=[0.0f32;LANES];
    for (i,&value) in x.iter().enumerate() {lanes[i%LANES]=step(fused,value,value,lanes[i%LANES]);}
    join(lanes)
}
/// Value `j` of a row of packed words: value `2i` is the low half of word `i`, value `2i+1` its high half.
fn packed(words:&[f32],j:usize)->f32 {
    let bits=words[j/2].to_bits();
    if j.is_multiple_of(2) {f32::from_bits(bits<<16)} else {f32::from_bits(bits&0xffff_0000)}
}
/// The host twin of an arm that has a product and then a sum; `None` for an arm that has none. Form 0 keeps the two
/// steps apart, and then the words must be those of the reference. Form 1 fuses each such pair. `GateBlend` adds two
/// products, so either can be the fused one: form 1 fuses the kept product, form 2 the message product.
fn twin(s:Shape,x:[&[f32];4],w:&[f32],form:u8)->Option<Vec<f32>> {
    let (h,d,t,k,r,fused)=(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,s.rows as usize,form!=0);
    // The first word of the row of the key or value plane at distance `dist` for a query head. The step words are
    // whole numbers here: the reference took them.
    let plane=|head:usize,dist:usize|((x[3][1] as usize+r-dist)%r*(h/t)+head/t)*(d/2);
    Some(match s.op {
        Op::HeadClip=>x[0].chunks_exact(d).enumerate().flat_map(|(head,row)|{
            let scaled=row.iter().enumerate().map(|(j,&value)|if t==1 {rn(value)*w[head*d+j]} else {rn(value)}).collect::<Vec<_>>();
            let reciprocal=1.0/(lane_squares(&scaled,fused).sqrt()+CLIP_EPSILON);
            let factor=s.bound*reciprocal;let factor=if factor>1.0 {1.0} else {factor};
            scaled.into_iter().map(move|value|{let clipped=rn(value*factor);if k==1 {rn(clipped*x[1][0])} else {clipped}})
        }).collect(),
        Op::RelBias=>(0..h*k).map(|index|{
            let (head,dist)=(index/k,index%k);
            let mut sum=0.0f32;for j in 0..d {sum=step(fused,rn(x[0][head*d+j]),w[j*k+dist],sum);}
            rn(sum)
        }).collect(),
        Op::WindowScores=>(0..h*r).map(|index|{
            let (head,dist)=(index/r,index%r);
            if dist>=x[3][0] as usize {return 0.0;}
            let key=&x[1][plane(head,dist)..];
            let mut sum=0.0f32;for j in 0..d {sum=step(fused,rn(x[0][head*d+j]),packed(key,j),sum);}
            let scaled=rn(rn(sum)*s.bound);
            rn(scaled+if dist<k {x[2][head*k+dist]} else {0.0})
        }).collect(),
        Op::WindowValues=>(0..h*d).map(|index|{
            let (head,channel,live)=(index/d,index%d,x[3][0] as usize);
            let (mut total,mut first)=(0.0f32,0);
            while first<live {
                let end=if live-first>BLOCK {first+BLOCK} else {live};
                let mut lanes=[0.0f32;LANES];
                for dist in first..end {lanes[dist%LANES]=step(fused,x[0][head*r+dist],packed(&x[1][plane(head,dist)..],channel),lanes[dist%LANES]);}
                total+=join(lanes);first=end;
            }
            rn(rn(total)*sigmoid(rn(x[2][head*d+channel])))
        }).collect(),
        Op::ShortConv=>{
            let (width,slot)=(h*d,x[3][0] as usize);
            (0..width).map(|index|{
                let mut sum=0.0f32;
                for lag in (0..t).rev() {
                    let mut tap=w[index*t+tap_place(lag,t)];
                    if s.cols&CONV_TOKEN_TAPS!=0 {tap+=rn(x[1][index/d*t+tap_place(lag,t)]);}
                    sum=step(fused,tap,rn(x[0][((slot+t-lag)%t)*width+index]),sum);
                }
                let y=rn((sum+rn(x[0][slot*width+index]))*s.bound);
                if s.cols&CONV_ADDEND==0 {y} else if s.cols&CONV_ROUND_SUM==0 {x[2][index]+y} else {rn(x[2][index]+y)}
            }).collect()
        },
        Op::MixBlend=>(0..h*d).map(|index|{
            let (query,channel)=(index/d,index%d);let dots=&x[0][query*k..(query+1)*k];
            let mut largest=f32::NEG_INFINITY;for &dot in dots {largest=largest.max(dot*s.bound);}
            let part=|dot:f32|exp(step(fused,dot,s.bound,-largest));
            let mut total=0.0f32;for &dot in dots {total+=part(dot);}
            let mut sum=0.0f32;for (i,&dot) in dots.iter().enumerate() {sum=step(fused,part(dot)/total,x[1][i*d+channel],sum);}
            sum
        }).collect(),
        Op::GateBlend=>(0..d).map(|i|{
            let (keep,gain)=(1.0-tanh(x[0][i]),tanh(x[0][d+i]));
            if form==2 {step(true,gain,x[2][i],keep*x[1][i])} else {step(fused,keep,x[1][i],gain*x[2][i])}
        }).collect(),
        Op::RouteFold=>(0..d).map(|index|{
            let mut sum=0.0f32;for (pick,&weight) in x[0].iter().enumerate() {sum=step(fused,rn(x[1][pick*d+index]),weight,sum);}
            rn(sum)
        }).collect(),
        Op::PlainNorm=>x[0].chunks_exact(d).flat_map(|row|{
            let scale=1.0/(lane_squares(row,fused)/d as f32+s.epsilon).sqrt();
            row.iter().map(move|&value|value*scale)
        }).collect(),
        Op::RoundedNorm=>x[0].chunks_exact(d).flat_map(|row|{
            let mut outer=[0.0f32;LANES];for (chunk,part) in row.chunks(128).enumerate() {outer[chunk%LANES]+=lane_squares(part,fused);}
            let scale=1.0/(join(outer)/d as f32+s.epsilon).sqrt();
            row.iter().enumerate().map(move|(j,&value)|if t==0 {rn(value*scale)} else {rn((value*scale)*w[j])})
        }).collect(),
        // One chain a dot: weight row `row`, input row `line`, the width in rising order from +0.0.
        Op::RowDots=>(0..r*h).map(|unit|{
            let (row,line)=(unit/h,unit%h);
            let mut sum=0.0f32;for j in 0..k {sum=step(fused,x[0][line*k+j],w[row*k+j],sum);}
            sum
        }).collect(),
        _=>return None,
    })
}
fn differing(a:&[f32],b:&[f32])->usize {a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count()}
/// Words, equal words, cases and launches of one arm, and what its twin showed.
#[derive(Clone,Copy,Default)]
struct Tally {words:usize,equal:usize,cases:usize,launches:usize,misses:usize,twin:bool,telling_cases:usize,telling_words:usize}
/// A device launch of one shape: the output words, or the refusal.
type Launch<'a>=&'a mut dyn FnMut(Shape,&[Vec<f32>;4],&[f32])->Result<Vec<f32>>;
fn save(path:&Path,text:&str)->Result<()> {
    fs::OpenOptions::new().create_new(true).write(true).open(path).and_then(|mut file|file.write_all(text.as_bytes())).map_err(|e|e.to_string())
}
/// The sweep. With no `launch` it is the host half alone.
fn sweep(out:&Path,mut launch:Option<Launch<'_>>)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let side=if launch.is_some() {"DEVICE"} else {"HOST"};
    let all=cases();let mut tally=[Tally::default();ARMS.len()];let (mut detail,mut notes,mut misses)=(String::new(),String::new(),String::new());
    let (mut dots_as_rows,mut dots_as_one)=(0usize,0usize);
    for case in &all {
        let arm=ARMS.iter().position(|(name,_)|*name==case.name).ok_or("a sweep case names no arm")?;
        tally[arm].cases+=1;let number=tally[arm].cases;
        for special in [false,true] {
            let (x,weight)=inputs(case,special)?;let refs=[&x[0][..],&x[1][..],&x[2][..],&x[3][..]];
            let expected=reference(case.shape,refs,&weight).map_err(|e|format!("{} {} special={special} host: {e}",case.name,case.label))?;
            // `RowDots` must give the words of two older operations of the current families: `ProjectRows` for 1 to 4
            // input rows in one launch, and `Project` for each input row alone.
            if case.shape.op==Op::RowDots {
                let (s,k)=(case.shape,case.shape.cols as usize);let (h,r)=(s.heads as usize,s.rows as usize);
                if h<=4 {
                    let rows=reference(Shape {op:Op::ProjectRows,..s},refs,&weight)?;
                    if rows.len()!=expected.len() || differing(&rows,&expected)!=0 {return Err(format!("RowDots {} special={special}: the words are not those of ProjectRows",case.label));}
                    dots_as_rows+=1;
                }
                for line in 0..h {
                    let one=reference(Shape {op:Op::Project,heads:0,..s},[&x[0][line*k..(line+1)*k],&[],&[],&[]],&weight)?;
                    if (0..r).any(|row|one[row].to_bits()!=expected[row*h+line].to_bits()) {return Err(format!("RowDots {} special={special}: input row {line} is not the words of Project",case.label));}
                    dots_as_one+=1;
                }
            }
            let mut telling=0;
            if let Some(apart)=twin(case.shape,refs,&weight,0) {
                if differing(&apart,&expected)!=0 || apart.len()!=expected.len() {return Err(format!("{} {} special={special}: the twin with its steps apart is not the reference",case.name,case.label));}
                telling=[1u8,2].iter().filter_map(|&form|twin(case.shape,refs,&weight,form)).map(|fused|differing(&fused,&expected)).min().unwrap_or(0);
                tally[arm].twin=true;tally[arm].telling_words+=telling;tally[arm].telling_cases+=usize::from(telling>0);
            }
            let Some(launch)=launch.as_mut() else {
                let _=writeln!(detail,"BLOCK-ARM HOST PASS arm={} case={number} words={} fused_words={telling} special={} {}",case.name,expected.len(),u8::from(special),case.label);
                continue;
            };
            for repeat in 1..=2 {
                let got=launch(case.shape,&x,&weight).map_err(|e|format!("{} {} special={special} device: {e}",case.name,case.label))?;
                if got.len()!=expected.len() {return Err(format!("{} {}: the device gave {} words",case.name,case.label,got.len()));}
                let equal=got.len()-differing(&got,&expected);
                tally[arm].words+=got.len();tally[arm].equal+=equal;tally[arm].launches+=1;
                if equal==got.len() {
                    let _=writeln!(detail,"BLOCK-ARM DEVICE PASS arm={} case={number} words={} fused_words={telling} special={} repeat={repeat} {}",case.name,got.len(),u8::from(special),case.label);
                } else {
                    let at=got.iter().zip(&expected).position(|(a,b)|a.to_bits()!=b.to_bits()).unwrap_or(0);
                    let line=format!("BLOCK-ARM DEVICE MISS arm={} case={number} word={at} host={:#010x} device={:#010x} differing_words={} special={} repeat={repeat} {}\n",
                        case.name,expected[at].to_bits(),got[at].to_bits(),got.len()-equal,u8::from(special),case.label);
                    detail.push_str(&line);tally[arm].misses+=1;
                    if tally[arm].misses<=6 {misses.push_str(&line);}
                }
            }
        }
    }
    // Refused shapes. An operand one word short: the host refuses, and the device refuses before a launch.
    let mut refused=0usize;
    for (name,probe) in ARMS {
        let Some((lengths,weights,_,_))=probe.lengths() else {continue};
        let case=Case {step:plain_step(probe.op),..case(name,"refused operand".into(),probe)};
        let (x,weight)=inputs(&case,false)?;
        for short in 0..5 {
            if short<4 && lengths[short]==0 || short==4 && weights==0 {continue;}
            let (mut x,mut weight)=(x.clone(),weight.clone());
            if short<4 {x[short].pop();} else {weight.pop();}
            if reference(probe,[&x[0],&x[1],&x[2],&x[3]],&weight).is_ok() {return Err(format!("{name}: the host takes operand {short} one word short"));}
            if let Some(launch)=launch.as_mut() && launch(probe,&x,&weight).is_ok() {return Err(format!("BLOCK-ARM DEVICE MISS arm={name}: the device takes operand {short} one word short"));}
            refused+=1;
        }
    }
    // Shapes with no lengths: a zero size, a missing cap or epsilon, and the form words that ShortConv does not have.
    let probe=|arm:usize|ARMS[arm].1;
    let mut bad=vec![Shape {bound:0.0,..probe(0)},Shape {taps:2,..probe(0)},Shape {cols:0,..probe(1)},Shape {taps:1,..probe(1)},
        Shape {cols:CONV_ROUND_SUM,..probe(2)},Shape {cols:CONV_ROUND_SUM|CONV_TOKEN_TAPS,..probe(2)},Shape {cols:2*CONV_ROUND_SUM,..probe(2)},Shape {bound:0.0,..probe(2)},
        Shape {dim:3,..probe(3)},Shape {rows:0,..probe(3)},Shape {bound:0.0,..probe(3)},Shape {dim:1,..probe(4)},Shape {rows:0,..probe(4)},Shape {cols:1,..probe(5)},Shape {taps:0,..probe(5)},
        Shape {bound:0.0,..probe(6)},Shape {cols:0,..probe(6)},Shape {dim:0,..probe(7)},Shape {epsilon:0.0,..probe(8)},Shape {bound:0.0,..probe(8)},Shape {rows:0,..probe(9)},
        Shape {bound:0.0,..probe(10)},Shape {epsilon:0.0,..probe(10)},Shape {epsilon:0.0,..probe(11)},Shape {heads:0,..probe(11)},Shape {epsilon:0.0,..probe(12)},Shape {taps:2,..probe(12)}];
    bad.retain(|shape|ARMS.iter().any(|(_,probe)|probe.op as u32==shape.op as u32 && probe.lengths().is_some()));
    for shape in &bad {
        if shape.lengths().is_some() {return Err(format!("the host gives lengths to the shape {shape:?}"));}
        let none:[Vec<f32>;4]=Default::default();
        if let Some(launch)=launch.as_mut() && launch(*shape,&none,&[]).is_ok() {return Err(format!("BLOCK-ARM DEVICE MISS: the device takes the shape {shape:?}"));}
        refused+=1;
    }
    let _=writeln!(notes,"BLOCK-ARM {side} NOTE refused: {refused} of {refused} (an operand one word short, or a shape with no lengths)");
    let _=writeln!(notes,"BLOCK-ARM {side} NOTE arm=RowDots the host words are those of ProjectRows in {dots_as_rows} of {dots_as_rows} launches of 1 to 4 input rows and those of Project in {dots_as_one} of {dots_as_one} input rows");
    // Step words that name no ring slot or no live count: the host refuses the result, and every word of the device
    // is a NaN.
    let wrong=[(2usize,vec![vec![4.0f32],vec![-1.0],vec![1.5]]),(3,vec![vec![5.0,0.0],vec![1.0,4.0],vec![1.5,0.0],vec![1.0,-1.0]]),
        (4,vec![vec![5.0],vec![1.5],vec![-1.0]]),(5,vec![vec![5.0,0.0],vec![1.0,4.0],vec![1.5,0.0],vec![1.0,-1.0]])];
    for (arm,steps) in wrong {
        let (name,probe)=ARMS[arm];if probe.lengths().is_none() {continue;}
        let mut line=String::new();
        for step in steps {
            let (x,weight)=inputs(&Case {step:step.clone(),..case(name,"wrong step words".into(),probe)},false)?;
            if reference(probe,[&x[0],&x[1],&x[2],&x[3]],&weight).is_ok() {return Err(format!("{name}: the host takes the wrong step words {step:?}"));}
            if let Some(launch)=launch.as_mut() {
                let got=launch(probe,&x,&weight)?;
                if got.is_empty() || got.iter().any(|v|!v.is_nan()) {return Err(format!("BLOCK-ARM DEVICE MISS arm={name} wrong step words {step:?}: the device gave a number"));}
                let _=write!(line," {step:?}:nan_words={}/{}",got.len(),got.len());
            } else {let _=write!(line," {step:?}:refused");}
        }
        let _=writeln!(notes,"BLOCK-ARM {side} NOTE arm={name} step words that name no slot or no live count:{line}");
    }
    // The same keys at two ring starts give the same words: the order of a sum over keys knows the distance only.
    for (arm,live) in [(3usize,256u32),(3,200),(5,256),(5,200)] {
        let (name,probe)=ARMS[arm];if probe.lengths().is_none() {continue;}
        let (rows,turn)=(256usize,100usize);
        let shape=Shape {heads:6,dim:16,taps:3,rows:rows as u32,cols:if arm==3 {8} else {0},..probe};
        let (x,weight)=inputs(&Case {step:vec![live as f32,0.0],..case(name,"two ring starts".into(),shape)},false)?;
        let mut turned=x.clone();turned[3][1]=turn as f32;let row=x[1].len()/rows;
        for at in 0..rows {let to=(at+turn)%rows;turned[1][to*row..(to+1)*row].copy_from_slice(&x[1][at*row..(at+1)*row]);}
        let (first,second)=(reference(shape,[&x[0],&x[1],&x[2],&x[3]],&weight)?,reference(shape,[&turned[0],&turned[1],&turned[2],&turned[3]],&weight)?);
        if differing(&first,&second)!=0 {return Err(format!("{name}: the two ring starts give other words on the host"));}
        if let Some(launch)=launch.as_mut() {
            for given in [&x,&turned] {
                let got=launch(shape,given,&weight)?;
                if got.len()!=first.len() || differing(&got,&first)!=0 {return Err(format!("BLOCK-ARM DEVICE MISS arm={name}: a ring start differs from the host"));}
            }
        }
        let _=writeln!(notes,"BLOCK-ARM {side} NOTE arm={name} two ring starts rows={rows} live={live}: {} of {} words equal at slot 0 and at slot {turn}",first.len(),first.len());
    }
    // RouteFold adds in the order of the picks. The same picks in the opposite order give other words where the sums
    // round in another way. The made data has the products 2^24, 1, 1, -2^24 in each channel: the sum is 0 in this
    // order and 2 in the opposite order. The second size has plain words.
    if ARMS[9].1.lengths().is_some() {
        for (picks,width,made) in [(4usize,33usize,true),(8,2304,false)] {
            let shape=Shape {rows:picks as u32,dim:width as u32,..empty(Op::RouteFold)};
            let (mut x,weight)=inputs(&Case {spans:[0.5,16.0,1.0,1.0],..case("RouteFold","two pick orders".into(),shape)},false)?;
            if made {
                x[0]=vec![1.0;picks];
                for (pick,value) in [16_777_216.0f32,1.0,1.0,-16_777_216.0].into_iter().enumerate() {x[1][pick*width..(pick+1)*width].fill(value);}
            }
            let mut turned=x.clone();turned[0].reverse();
            for pick in 0..picks {turned[1][pick*width..(pick+1)*width].copy_from_slice(&x[1][(picks-1-pick)*width..(picks-pick)*width]);}
            let (first,second)=(reference(shape,[&x[0],&x[1],&x[2],&x[3]],&weight)?,reference(shape,[&turned[0],&turned[1],&turned[2],&turned[3]],&weight)?);
            let other=differing(&first,&second);
            if made && other!=first.len() {return Err("RouteFold: the made data gives the same word in the two pick orders".into());}
            if let Some(launch)=launch.as_mut() {
                for (given,expected) in [(&x,&first),(&turned,&second)] {
                    let got=launch(shape,given,&weight)?;
                    if got.len()!=expected.len() || differing(&got,expected)!=0 {return Err("BLOCK-ARM DEVICE MISS arm=RouteFold: a pick order differs from the host".into());}
                }
            }
            let _=writeln!(notes,"BLOCK-ARM {side} NOTE arm=RouteFold two pick orders picks={picks} width={width} made_data={}: words_that_differ={other} of {}",u8::from(made),first.len());
        }
    }
    let mut receipt=String::new();let (mut swept,mut words,mut differing_words,mut pending,mut blind,mut untwinned)=(0usize,0usize,0usize,0usize,0usize,0usize);
    for (arm,(name,probe)) in ARMS.iter().enumerate() {
        let t=tally[arm];
        if t.cases>0 {
            swept+=1;words+=t.words;differing_words+=t.words-t.equal;
            let fused=if t.twin {format!("fused_telling_runs={} fused_telling_words={}",t.telling_cases,t.telling_words)}
                else if matches!(probe.op,Op::CappedGate|Op::Softcap|Op::SinkSoftmax) {"fused_step=none (no sum after a product)".to_string()}
                else {untwinned+=1;"fused_step=not checked (no twin in this build)".to_string()};
            if t.twin && t.telling_cases==0 {blind+=1;}
            let _=writeln!(receipt,"BLOCK ARM {side} {name} words_equal={} words={} cases={} launches={} {fused}",t.equal,t.words,t.cases,t.launches);
        } else if probe.lengths().is_some() {
            pending+=1;let _=writeln!(receipt,"BLOCK ARM {side} {name} is written and has no sweep case in this build");
        } else {let _=writeln!(receipt,"BLOCK ARM {side} {name} not written at this head (no lengths)");}
    }
    receipt.push_str(&misses);receipt.push_str(&notes);
    let pass=differing_words==0 && pending==0 && blind==0;
    let _=writeln!(receipt,"BLOCK ARMS {side} {} arms_swept={swept} arms_written_without_case={pending} arms_with_no_fused_telling_case={blind} arms_without_twin={untwinned} cases={} words={words} differing_words={differing_words} reference=causal_ops::reference repeats=2 signed_zeros=1 subnormals=1 bf16_ties=1",
        if pass {"PASS"} else {"FAIL"},all.len());
    save(&out.join("CASES.txt"),&detail)?;save(&out.join("RESULT.txt"),&receipt)?;
    print!("{receipt}");
    if !pass {return Err(format!("block arm sweep: {differing_words} differing words, {pending} written arms with no case, {blind} arms with no case that tells a fused step"));}
    save(&out.join("COMPLETE"),&receipt)
}
/// The host half alone: the cases, the reference, the twins and the refused shapes.
pub fn host(out:&Path)->Result<()> {sweep(out,None)}
/// Mode `block-arms-entries`: one direct launch of each operation of the general kernel that the token step of the
/// loop family uses beside the block arms, against the host reference. Today that is the router pick, small and in
/// its ranked layout, which is one block of many threads. The sweep of the arms launches none of them, so a change
/// of the general kernel that keeps every arm word can still stop the token step at such a launch.
/// Each case has its own backend: a refused launch must not reach the next case.
#[cfg(target_os="linux")]
pub fn entries(out:&Path)->Result<()> {
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut text,mut failed)=(String::new(),0);
    for (rows,picks) in [(8u32,2u32),(128,8),(512,8)] {
        let shape=Shape {op:Op::Route,heads:1,dim:1,taps:1,rows,cols:picks,epsilon:0.0,bound:8.0};
        let (lengths,weights,units,_)=shape.lengths().ok_or("the route shape has no lengths")?;
        let x:[Vec<f32>;4]=core::array::from_fn(|i|fill(lengths[i],i+1,false,1.0));let weight=fill(weights,5,false,0.25);
        let expected=reference(shape,[&x[0][..],&x[1][..],&x[2][..],&x[3][..]],&weight)?;
        let (blocks,threads)=spark_gpu_probe::causal_ops::geometry(shape,units);
        let got=spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            let mut device=Vec::new();
            for input in &x {device.push(if input.is_empty() {None} else {Some(b.upload_vector(input)?)});}
            let device_weight=if weight.is_empty() {None} else {Some(b.upload_vector(&weight)?)};
            let result=b.causal_operation(shape,[device[0].as_ref(),device[1].as_ref(),device[2].as_ref(),device[3].as_ref()],device_weight.as_ref())?;
            b.read_vector_values(&result)
        });
        let state=match got {
            Ok(got) if got.len()==expected.len() && differing(&got,&expected)==0=>format!("words={} equal",got.len()),
            Ok(got)=>{failed+=1;format!("words={} DIFFER differing_words={}",got.len(),differing(&got,&expected))},
            Err(error)=>{failed+=1;format!("LAUNCH REFUSED error={error}")},
        };
        let line=format!("BLOCK-ARM ENTRY op=Route rows={rows} picks={picks} ranked={} blocks={blocks} threads={threads} {state}",spark_gpu_probe::latent_ops::route_ranked(shape));
        eprintln!("{line}");text.push_str(&line);text.push('\n');
    }
    let write=|name:&str,text:&str|fs::write(out.join(name),text).map_err(|e|format!("{}: {e}",out.join(name).display()));
    write("ENTRIES.txt",&text)?;
    if failed>0 {return Err(format!("{failed} of the entry launches did not pass; see ENTRIES.txt"));}
    write("COMPLETE","")
}
/// The device sweep on GPU 0.
#[cfg(target_os="linux")]
pub fn run(out:&Path)->Result<()> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        let mut launch=|shape:Shape,x:&[Vec<f32>;4],weight:&[f32]|->Result<Vec<f32>> {
            let mut device=Vec::new();
            for input in x {device.push(if input.is_empty() {None} else {Some(b.upload_vector(input)?)});}
            let device_weight=if weight.is_empty() {None} else {Some(b.upload_vector(weight)?)};
            let result=b.causal_operation(shape,[device[0].as_ref(),device[1].as_ref(),device[2].as_ref(),device[3].as_ref()],device_weight.as_ref())?;
            b.read_vector_values(&result)
        };
        sweep(out,Some(&mut launch))
    })
}
/// The time of one launch of each shape in a list, on GPU 0. A line of the list holds a label, the arm, the 5 counts
/// and the 2 numbers of the shape, the count of such launches in one token, and the step words of the arm. A line
/// that starts with `#` is a note. The sizes come from the list, so this code holds no size of a model.
/// Each shape runs `TIMED` times after `WARM` launches that are not timed. One time is from the call of the launch to
/// the end of the device work. The first result is compared with the host, so the timed launch is the right one.
#[cfg(target_os="linux")]
pub fn time(out:&Path,list:&Path)->Result<()> {
    const WARM:usize=8;
    const TIMED:usize=200;
    fs::create_dir(out).map_err(|e|e.to_string())?;
    let text=fs::read_to_string(list).map_err(|e|e.to_string())?;
    let mut receipt=String::new();let (mut token_micros,mut token_launches)=(0.0f64,0.0f64);
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for line in text.lines().filter(|line|!line.trim().is_empty() && !line.starts_with('#')) {
            let words=line.split_whitespace().collect::<Vec<_>>();
            if words.len()<10 {return Err(format!("arm time list: a line needs 10 words or more: {line}"));}
            let (name,probe)=*ARMS.iter().find(|(name,_)|*name==words[1]).ok_or_else(||format!("arm time list: no arm {}",words[1]))?;
            let count=|at:usize|words[at].parse::<u32>().map_err(|e|format!("arm time list: {}: {e}",words[at]));
            let number=|at:usize|words[at].parse::<f32>().map_err(|e|format!("arm time list: {}: {e}",words[at]));
            let shape=Shape {op:probe.op,heads:count(2)?,dim:count(3)?,taps:count(4)?,rows:count(5)?,cols:count(6)?,epsilon:number(7)?,bound:number(8)?};
            let a_token=words[9].parse::<f64>().map_err(|e|format!("arm time list: {}: {e}",words[9]))?;
            let step=(10..words.len()).map(number).collect::<Result<Vec<_>>>()?;
            let (x,weight)=inputs(&Case {step,..case(name,words[0].to_string(),shape)},false)?;
            let expected=reference(shape,[&x[0],&x[1],&x[2],&x[3]],&weight).map_err(|e|format!("{} host: {e}",words[0]))?;
            let mut device=Vec::new();
            for input in &x {device.push(if input.is_empty() {None} else {Some(b.upload_vector(input)?)});}
            let device_weight=if weight.is_empty() {None} else {Some(b.upload_vector(&weight)?)};
            let operands=[device[0].as_ref(),device[1].as_ref(),device[2].as_ref(),device[3].as_ref()];
            let first=b.causal_operation(shape,operands,device_weight.as_ref()).map_err(|e|format!("{} device: {e}",words[0]))?;
            let got=b.read_vector_values(&first)?;
            if got.len()!=expected.len() || differing(&got,&expected)!=0 {return Err(format!("BLOCK-ARM TIME MISS {}: the device words are not the host words",words[0]));}
            drop(first);
            let mut micros=Vec::with_capacity(TIMED);
            for turn in 0..WARM+TIMED {
                b.synchronize()?;
                let start=std::time::Instant::now();
                let result=b.causal_operation(shape,operands,device_weight.as_ref())?;
                b.synchronize()?;
                let spent=start.elapsed().as_secs_f64()*1.0e6;
                drop(result);
                if turn>=WARM {micros.push(spent);}
            }
            micros.sort_by(f64::total_cmp);
            let mean=micros.iter().sum::<f64>()/TIMED as f64;
            token_micros+=mean*a_token;token_launches+=a_token;
            let _=writeln!(receipt,"BLOCK-ARM TIME arm={name} label={} heads={} dim={} taps={} rows={} cols={} words={} launches={TIMED} mean_us={mean:.1} min_us={:.1} median_us={:.1} max_us={:.1} a_token={a_token} host_equal=1",
                words[0],shape.heads,shape.dim,shape.taps,shape.rows,shape.cols,expected.len(),micros[0],micros[TIMED/2],micros[TIMED-1]);
        }
        Ok(())
    })?;
    let _=writeln!(receipt,"BLOCK ARMS TIME token_arm_ms={:.3} arm_launches_a_token={token_launches} (the mean time of a launch times its count in one token, summed; the row kernels, the copies and the reads are not in it)",token_micros/1000.0);
    save(&out.join("RESULT.txt"),&receipt)?;
    print!("{receipt}");
    save(&out.join("COMPLETE"),&receipt)
}
