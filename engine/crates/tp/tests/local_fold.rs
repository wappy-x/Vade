//! LOCAL FOLD twin: fold-then-gather against gather-then-fold on the host reference of the fold (no device).
use spark_gpu_probe::{bf16::rounded,causal_ops::{reference,Op,Shape}};

/// LOCAL FOLD (tp resident_weights apply_routed_fold): each rank rounds and folds its own rows of the picked experts'
/// down outputs, and the folded rows are gathered. Every word must have the bits of the fold of the gathered rows.
#[test]
fn expert_fold_of_each_ranks_rows_equals_the_fold_of_the_gathered_rows() {
    // Words that test the fold's own rounding: signed zeros, the smallest values, BF16 ties and their neighbours, large values.
    const SPECIAL:[u32;16]=[0,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_8000,0x3f80_7fff,0x3f81_8000,0xbf80_8000,
        0x3f7f_ffff,0x4000_0001,0xc0ff_ffff,0x477f_e000,0xc77f_e000,0x3380_0000,0xb380_0000];
    let value=|seed:usize,case:usize|if case>0 && seed%3!=1 {f32::from_bits(SPECIAL[(seed/3+case)%SPECIAL.len()])} else {((seed*2654435761usize)%2001) as f32/250.0-4.0};
    for (picks,hidden,ranks) in [(8usize,4096usize,2usize),(8,4096,4),(8,4096,8),(1,64,2),(5,384,3),(32,128,2)] {
        for taps in [0u32,1] {for case in 0..3 {
            let ids=(0..picks).map(|pick|((pick*37+11+case*5)%288) as f32).collect::<Vec<_>>();
            let weights=(0..picks).map(|pick|rounded(if case==2 && pick%3==0 {-0.0} else {value(pick*7+case,case)/8.0})).collect::<Vec<_>>();
            let raw=(0..picks*hidden).map(|word|value(word+17*case,case)).collect::<Vec<_>>();
            let shared=if taps!=0 {(0..hidden).map(|word|rounded(value(word*5+3,case))).collect::<Vec<_>>()} else {Vec::new()};
            // Gather, then fold: the joined put rounds each rank's rows, the fold reads the gathered rows.
            let whole=Shape {op:Op::ExpertFold,rows:picks as u32,cols:hidden as u32,taps,heads:0,dim:0,epsilon:0.0,bound:0.0};
            let gathered=raw.iter().map(|&word|rounded(word)).collect::<Vec<_>>();
            let expected=reference(whole,[&ids,&weights,&gathered,&shared],&[]).unwrap();
            // Fold, then gather: each rank rounds and folds its own rows.
            let count=hidden/ranks;let mut got=vec![f32::NAN;hidden];
            for rank in 0..ranks {
                let start=rank*count;
                let own=(0..picks).flat_map(|pick|raw[pick*hidden+start..pick*hidden+start+count].iter().map(|&word|rounded(word))).collect::<Vec<_>>();
                let own_shared=if taps!=0 {&shared[start..start+count]} else {&shared[..]};
                let part=reference(Shape {cols:count as u32,..whole},[&ids,&weights,&own,own_shared],&[]).unwrap();
                got[start..start+count].copy_from_slice(&part);
            }
            assert!(expected.iter().map(|v|v.to_bits()).eq(got.iter().map(|v|v.to_bits())),"picks={picks} hidden={hidden} ranks={ranks} taps={taps} case={case}");
            assert!(case==0 || raw.iter().any(|v|v.to_bits()==0x8000_0000) && raw.iter().any(|v|v.to_bits()==0x3f80_8000),
                "the special case holds no negative zero or no tie: picks={picks} hidden={hidden}");
        }}
    }
}
