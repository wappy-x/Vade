use spark_gpu_probe::mhc_boundary::{self as boundary,Span};

#[test]
fn fused_boundary_abi_and_norm_chunks_cover_all_features_once() {
    use spark_gpu_probe::mhc_params::{Boundary,Shape};
    assert_eq!(std::mem::size_of::<Boundary>(),136);assert_eq!(std::mem::align_of::<Boundary>(),8);
    for streams in [1,3,4,7,33] {for hidden in [1usize,3,31,32,129,4096,4103] {
        let shape=Shape {rows:3,streams,hidden:hidden as u32,iterations:20,rms_eps:1e-6,eps:1e-6};shape.validate().unwrap();
        let warps=shape.prepare_threads() as usize/32;let chunks=hidden.div_ceil(128);let mut visits=vec![0;hidden];let mut partials=vec![0;chunks];
        for warp in 0..warps {for chunk in (warp..chunks).step_by(warps) {
            partials[chunk]+=1;
            for lane in 0..32 {for feature in (chunk*128+lane..((chunk+1)*128).min(hidden)).step_by(32) {visits[feature]+=1;}}
        }}
        assert!(visits.iter().all(|&count|count==1));assert!(partials.iter().all(|&count|count==1));
    }}
}

#[test]
fn boundary_warps_cover_each_complete_coefficient_in_row_major_storage() {
    use spark_gpu_probe::mhc_params::Shape;
    for rows in [1,3] {for streams in [1,4,7,33] {
        let shape=Shape {rows,streams,hidden:7,iterations:20,rms_eps:1e-6,eps:1e-6};shape.validate().unwrap();
        let count=shape.coefficients();let mut coefficients=vec![0;rows as usize*count];let warps=shape.prepare_threads() as usize/32;
        for token in 0..rows as usize {for warp in 0..warps {for coefficient in (warp..count).step_by(warps) {
            coefficients[token*count+coefficient]+=1;
        }
        }}
        assert!(coefficients.iter().all(|&count|count==1));
    }}
}

#[test]
fn prepare_blocks_cover_every_independent_coefficient_with_complete_warps() {
    use spark_gpu_probe::mhc_params::{Shape,Prepare};
    assert_eq!(std::mem::size_of::<Prepare>(),80);
    for streams in 1..=65 {
        let shape=Shape {rows:3,streams,hidden:7,iterations:20,rms_eps:1e-6,eps:1e-6};shape.validate().unwrap();
        let threads=shape.prepare_threads() as usize;assert!(threads>0 && threads<=1024 && threads%32==0);
        let warps=threads/32;let mut visits=vec![0;shape.coefficients()];
        for warp in 0..warps {for coefficient in (warp..visits.len()).step_by(warps) {visits[coefficient]+=1;}}
        assert!(visits.iter().all(|&count|count==1));
        for hidden in [1,7,31,32,129,4096] {
            let mut collapsed=vec![0;hidden];
            for thread in 32..threads {for feature in ((thread-32)..hidden).step_by(threads-32) {collapsed[feature]+=1;}}
            assert!(collapsed.iter().all(|&count|count==1));
        }
    }
}

#[test]
fn complete_warp_broadcasts_visit_each_feature_in_order_without_tail_terms() {
    for length in [1,3,31,32,33,97] {
        let values=(0..length).map(|index| [16777216.0,1.0,-16777216.0,-0.0][index%4]).collect::<Vec<f32>>();
        let expected=values.iter().fold(0.0f32,|sum,value|sum+value);
        for lane in 0..32 {
            let mut actual=0.0f32;
            for first in (0..length).step_by(32) {for source in 0..32 {
                let from=lane^(lane^source);assert_eq!(from,source);
                if first+source<length {actual+=values[first+from];}
            }}
            assert_eq!(actual.to_bits(),expected.to_bits());
        }
    }
}

#[test]
fn collapse_rounds_the_completed_fp32_sum() {
    // A premature BF16 coefficient cast would produce 1.0 here.
    assert_eq!(boundary::collapse(&[1.0,1.0],&[1.00390625,0.00390625],2,1,0).to_bits(),1.0078125f32.to_bits());
    assert_eq!(boundary::collapse(&[1.0],&[1.00390625],1,1,0).to_bits(),1.0f32.to_bits());
    assert_eq!(boundary::collapse(&[1.0],&[1.01171875],1,1,0).to_bits(),1.015625f32.to_bits());
}
#[test]
fn combine_casts_coefficients_and_each_term_before_the_final_add() {
    let coefficients=[0.0,0.0,1.0,1.0,1.0,1.0,0.00390625,0.00390625];
    // The residual term is a BF16 tie. A single final cast would give 1.0078125.
    for destination in 0..2 {
        assert_eq!(boundary::combine(&[1.0,1.0],&coefficients,0.00390625,2,1,destination,0),1.0);
    }
    // Post must round before its multiply. A final-only cast gives 1.5078125.
    let coefficients=[0.0,1.00390625,0.0];
    assert_eq!(boundary::combine(&[0.0],&coefficients,1.5,1,1,0,0),1.5);
}
#[test]
fn combine_uses_source_destination_axes_and_global_stream_order() {
    let c=[0.0,0.0,1.0,2.0,1.0,2.0,3.0,4.0];let x=[1.0,2.0,3.0,4.0];let block=[1.0,2.0];
    let actual=(0..2).flat_map(|dst| (0..2).map(move |d| boundary::combine(&x,&c,block[d],2,2,dst,d))).collect::<Vec<_>>();
    assert_eq!(actual,[11.0,16.0,16.0,24.0]);
}
#[test]
fn bf16_lane_upload_rejects_lossy_and_nonfinite_values() {
    boundary::check_bf16(&[0.0,-0.0,1.0,-1.0,f32::from_bits(0x00010000),f32::from_bits(0x7f7f0000)]).unwrap();
    for values in [&[][..],&[1.00390625],&[f32::NAN],&[f32::INFINITY],&[f32::from_bits(1)]] {
        assert!(boundary::check_bf16(values).is_err());
    }
}
#[test]
fn resident_spans_reject_aliases_overflow_and_wrong_extents() {
    let spans=[Span {address:4096,words:2},Span {address:4104,words:3},Span {address:4116,words:1}];
    boundary::check_spans(&spans,&[2,3,1]).unwrap();
    assert!(boundary::check_spans(&spans,&[2,3]).is_err());
    for i in 0..spans.len() {for case in 0..6 {
        let mut bad=spans;
        match case {0=>bad[i].address=0,1=>bad[i].address+=1,2=>bad[i].words-=1,3=>bad[i].words+=1,
            4=>bad[i].address=u64::MAX-3,_=>bad[i].words=usize::MAX}
        assert!(boundary::check_spans(&bad,&[2,3,1]).is_err());
    }}
    for i in 0..spans.len() {for j in 0..i {
        let mut bad=spans;bad[i].address=bad[j].address;assert!(boundary::check_spans(&bad,&[2,3,1]).is_err());
    }}
}

#[test]
fn grouped_square_loads_keep_the_full_ascending_rms_fold_at_width_tails() {
    use spark_gpu_probe::{mhc_params::Shape,mhc_reference};
    for streams in [1u32,2,4] {for hidden in [1u32,2,7,8,9,17,128,515] {
        let shape=Shape {rows:1,streams,hidden,iterations:4,rms_eps:1e-5,eps:1e-5};shape.validate().unwrap();
        let x=(0..shape.wide()).map(|i|if i%13==0 {-0.0} else {(i as i32%31-15) as f32*0.037}).collect::<Vec<_>>();
        let mut squares=0.0f32;for &v in &x {squares+=v*v;}
        let inv=1.0/(squares/shape.wide() as f32+shape.rms_eps).sqrt();
        let expected=x.iter().map(|&v|(v*inv).to_bits()).collect::<Vec<_>>();let mut got=vec![f32::NAN;x.len()];
        mhc_reference::normalize(shape,&x,&mut got);
        assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected,"streams={streams} hidden={hidden}");
    }}
}
