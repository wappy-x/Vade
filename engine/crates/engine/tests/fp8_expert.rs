#[allow(dead_code)] #[path = "../../../tools/bench/src/fp8_reference.rs"] mod reference;
use spark_gpu_probe::fp8_abi::{ExpertProjection, RoutingLayout};
#[test]
fn fixed_expert_rows() {
    for (h,i) in [(2048usize,512usize),(3072,1024)] { for k in [h,i,i/2] {
        let l=RoutingLayout::checked(8,i as u32).unwrap(); for (n,c) in [(i,h),(i/2,h),(h,i),(h,i/2)] { assert!(ExpertProjection::checked(256,512,n as u64,c as u64,2,3).is_some()); }
        let row:Vec<_>=(0..k).map(|c| reference::round_bf16(((c%31) as f32-15.0)/16.0)).collect(); let solo=reference::quantize(&row,k).unwrap(); let weights=vec![56;k*128]; let scales=vec![0x3f80;k/128]; let expected=reference::block_gemm(&solo,&weights,&scales,128,0..k/128).unwrap();
        for m in [1usize,2,8,16,64,256] { let mut input=vec![0.0;m*k]; for r in 0..m { for c in 0..k { input[r*k+c]=if r==m/2 {row[c]} else {((r+c)%17) as f32/8.0}; } } let q=reference::quantize(&input,k).unwrap();
            assert_eq!(&q.codes[m/2*k..(m/2+1)*k],solo.codes); assert_eq!(q.scales[m/2*k/128..(m/2+1)*k/128].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),solo.scales.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let selected=reference::Quantized{rows:1,cols:k,codes:q.codes[m/2*k..(m/2+1)*k].to_vec(),scales:q.scales[m/2*k/128..(m/2+1)*k/128].to_vec()}; assert_eq!(reference::block_gemm(&selected,&weights,&scales,128,0..k/128).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>()); if k<=i { assert_eq!(l.code_index(16,m/2,k,0),(16+m/2)*i); assert_eq!(l.scale_index(16,m/2,k,0),(16+m/2)*i/128); }
        }
    } }
}

// Same H=I=128, input rows and constant FP8 expert weights as fp8_routed_gpu.
fn routed_expert_reference(id: usize, expert: usize) -> Vec<u32> {
    let input: Vec<_> = (0..128).map(|col| ((id*13+col)%31+1) as f32/32.0).collect();
    let q = reference::quantize(&input,128).unwrap();
    let weights = vec![0x30+(expert%8) as u8;128*128];
    let gate = reference::block_gemm(&q,&weights,&[0x3c00],128,0..1).unwrap();
    let middle: Vec<_> = gate.into_iter().map(|g| {
        let g = reference::round_bf16(g);
        reference::round_bf16(reference::round_bf16(g/(1.0+(-g).exp()))*g)
    }).collect();
    reference::block_gemm(&reference::quantize(&middle,128).unwrap(),&weights,&[0x3c00],128,0..1)
        .unwrap().into_iter().map(f32::to_bits).collect()
}

#[test]
fn routed_rejected_compaction_reference() {
    let values: Vec<_> = (0..31).map(|id| (0..8).map(|e| routed_expert_reference(id,e)).collect::<Vec<_>>()).collect();
    let batch = |ids: &[usize], k: usize, bad: Option<usize>, cutoff: usize, legacy: bool, sorted: bool| {
        let layout = RoutingLayout::checked(k as u32,128).unwrap();
        ids.iter().enumerate().map(|(row,&id)| {
            let mut output = vec![0x7fc00123;(k+1)*128];
            if bad == Some(row) { output[..k*128].fill(0); return output; }
            let mut experts: Vec<_> = (0..k).map(|lane| ((id*7+lane*3)%32) as u32).collect();
            if sorted { experts.sort_unstable(); }
            // Independent general-path reference: sort by (expert ID, pair index).
            let mut order: Vec<_> = (0..k).collect();
            order.sort_by_key(|&lane| (experts[lane],lane));
            for (rank,&lane) in order.iter().enumerate() {
                let slot = if ids.len() > cutoff { rank } else if legacy { lane }
                    else { layout.ordered_slot(lane,|at| experts[at]) };
                output[slot*128..(slot+1)*128].copy_from_slice(&values[id%31][experts[lane] as usize%8]);
            }
            output
        }).collect::<Vec<_>>()
    };
    let mut old_failures = Vec::new();
    for cutoff in [1,4] { for k in [1,8,32] { for t in [1,2,3,4,5,17,256] {
        let ids: Vec<_> = (0..t).collect();
        let bad = t/2;
        let healthy: Vec<_> = ids.iter().copied().filter(|&id| id != bad).collect();
        let poisoned = batch(&ids,k,Some(bad),cutoff,false,false);
        let clean = batch(&ids,k,None,cutoff,false,false);
        let compact = batch(&healthy,k,None,cutoff,false,false);
        assert!(poisoned[bad][..k*128].iter().all(|&v| v == 0));
        assert!(poisoned[bad][k*128..].iter().all(|&v| v == 0x7fc00123));
        for (row,&id) in healthy.iter().enumerate() {
            assert_eq!(poisoned[id],clean[id]);
            assert_eq!(poisoned[id],compact[row],"cutoff={cutoff} t={t} k={k} id={id}");
        }
        let old = batch(&ids,k,Some(bad),cutoff,true,false);
        let old_compact = batch(&healthy,k,None,cutoff,true,false);
        if healthy.iter().enumerate().any(|(row,&id)| old[id] != old_compact[row]) {
            old_failures.push((cutoff,t,k));
        }
        // Serving routes already arrive sorted: all no-rejection bytes stay fixed.
        assert_eq!(batch(&ids,k,None,cutoff,false,true),batch(&ids,k,None,cutoff,true,true));
    } } }
    assert_eq!(old_failures,[(1,2,32),(4,5,8),(4,5,32)]);
}

#[test]
fn routed_slots_ties_and_shared() {
    let experts = [7,2,7,1,2];
    let layout = RoutingLayout::checked(5,128).unwrap();
    assert_eq!((0..5).map(|lane| layout.ordered_slot(lane,|at| experts[at])).collect::<Vec<_>>(),[3,1,4,0,2]);
    assert_eq!(layout.ordered_slot(5,|_| panic!("shared slot must not read routes")),5);
}
