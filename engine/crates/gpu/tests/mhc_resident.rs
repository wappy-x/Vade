use spark_gpu_probe::mhc_boundary::{self as boundary,Span};
use spark_gpu_probe::mhc_params;
#[allow(dead_code)]
#[path="../kernels/src/qsa_common.rs"] mod qsa_common;
#[allow(dead_code)]
#[allow(clippy::chunks_exact_to_as_chunks)] // Shared with the NVPTX crate; fix it with the kernels.
#[path="../kernels/src/mhc_math.rs"] mod mhc_math;

#[test]
fn warp_sinkhorn_preserves_order_and_uses_only_live_shuffle_sources() {
    for streams in 1usize..=5 {for iterations in [1,2,20] {
        let shape=mhc_params::Shape {rows:1,streams:streams as u32,hidden:7,iterations,rms_eps:1e-6,eps:1e-6};
        let mut reference=(0..streams*streams).map(|index|match index%7 {0=>-0.0,1=>0.0,2=>100.0,3=>-100.0,_=>(index as i32%19-9) as f32/7.0}).collect::<Vec<_>>();
        let mut lanes=std::array::from_fn::<_,32,_>(|lane|reference.get(lane).copied().unwrap_or(0.0));
        let source=|lane:usize,index:usize,values:&[f32;32]| {
            assert!(index<streams*streams);
            assert_eq!(lane^(lane^index),index);
            values[index]
        };
        lanes=std::array::from_fn(|lane| {
            let row=(lane/streams).min(streams-1);let mut maximum=f32::NEG_INFINITY;
            for column in 0..streams {maximum=maximum.max(source(lane,row*streams+column,&lanes));}
            qsa_common::exponential(lanes[lane]-maximum)
        });
        lanes=std::array::from_fn(|lane| {
            let row=(lane/streams).min(streams-1);let mut sum=0.0f32;
            for column in 0..streams {sum+=source(lane,row*streams+column,&lanes);}
            lanes[lane]/sum+shape.eps
        });
        for iteration in 0..iterations {
            if iteration>0 {lanes=std::array::from_fn(|lane| {
                let row=(lane/streams).min(streams-1);let mut sum=0.0f32;
                for column in 0..streams {sum+=source(lane,row*streams+column,&lanes);}
                lanes[lane]/(sum+shape.eps)
            });}
            lanes=std::array::from_fn(|lane| {
                let column=lane%streams;let mut sum=0.0f32;
                for row in 0..streams {sum+=source(lane,row*streams+column,&lanes);}
                lanes[lane]/(sum+shape.eps)
            });
        }
        mhc_math::sinkhorn(shape,&mut reference);
        assert_eq!(lanes[..streams*streams].iter().map(|value|value.to_bits()).collect::<Vec<_>>(),reference.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
    }}
}

#[test]
fn paired_register_fold_loads_and_adds_each_slot_word_once() {
    for span in 0usize..=1024 {
        let full=span/64*64;let mut loads=Vec::new();let mut adds=Vec::new();
        if full>0 {
            loads.extend(0..32);let mut cursor=32;let mut first=0;
            loop {
                loads.extend(cursor..cursor+32);let second=cursor;cursor+=32;
                let more=cursor<full;adds.extend(first..first+32);
                if more {first=cursor;loads.extend(cursor..cursor+32);cursor+=32;}
                adds.extend(second..second+32);
                if !more {break;}
            }
        }
        loads.extend(full..span);adds.extend(full..span);
        assert_eq!(loads,(0..span).collect::<Vec<_>>());assert_eq!(adds,loads);
    }
}

#[test]
fn shared_sinkhorn_phase_order_matches_every_serial_row_and_column_fold() {
    for streams in [1usize,2,3,4,5,8,31] {for iterations in [1,2,20] {
        let shape=mhc_params::Shape {rows:1,streams:streams as u32,hidden:7,iterations,rms_eps:1e-6,eps:1e-6};
        let mut reference=(0..shape.coefficients()).map(|index|match index%7 {0=>-0.0,1=>0.0,2=>100.0,3=>-100.0,_=>(index as i32%19-9) as f32/7.0}).collect::<Vec<_>>();
        let mut actual=reference.clone();mhc_math::activate(shape,&mut reference);
        for stream in 0..streams {actual[stream]=mhc_math::sigmoid(actual[stream])+shape.eps;actual[streams+stream]=2.0*mhc_math::sigmoid(actual[streams+stream]);}
        let matrix=&mut actual[2*streams..];
        let maxima=(0..streams).map(|row|matrix[row*streams..(row+1)*streams].iter().copied().fold(f32::NEG_INFINITY,f32::max)).collect::<Vec<_>>();
        for cell in 0..matrix.len() {matrix[cell]=qsa_common::exponential(matrix[cell]-maxima[cell/streams]);}
        let sums=(0..streams).map(|row|matrix[row*streams..(row+1)*streams].iter().copied().fold(0.0f32,|sum,value|sum+value)).collect::<Vec<_>>();
        for cell in 0..matrix.len() {matrix[cell]=matrix[cell]/sums[cell/streams]+shape.eps;}
        for iteration in 0..iterations {
            if iteration>0 {
                let sums=(0..streams).map(|row|matrix[row*streams..(row+1)*streams].iter().copied().fold(0.0f32,|sum,value|sum+value)+shape.eps).collect::<Vec<_>>();
                for cell in 0..matrix.len() {matrix[cell]/=sums[cell/streams];}
            }
            let sums=(0..streams).map(|column|(0..streams).fold(0.0f32,|sum,row|sum+matrix[row*streams+column])+shape.eps).collect::<Vec<_>>();
            for cell in 0..matrix.len() {matrix[cell]/=sums[cell%streams];}
        }
        assert_eq!(actual.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),reference.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
        for threads in [32usize,64,128,768,1024] {
            let mut visits=vec![0;streams*streams];
            for thread in 0..threads {for cell in (thread..visits.len()).step_by(threads) {visits[cell]+=1;}}
            assert!(visits.iter().all(|&count|count==1));
        }
    }}
}

#[test]
fn fused_boundary_abi_and_norm_chunks_cover_all_features_once() {
    use spark_gpu_probe::mhc_params::{Boundary,Shape};
    assert_eq!(std::mem::size_of::<Boundary>(),160);assert_eq!(std::mem::align_of::<Boundary>(),8);
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
        let threads=shape.prepare_threads() as usize;assert!(threads>0 && threads<=1024 && threads.is_multiple_of(32));
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
#[allow(clippy::excessive_precision)] // literal constants stay as written.
fn collapse_rounds_the_completed_fp32_sum() {
    // A premature BF16 coefficient cast would produce 1.0 here.
    assert_eq!(boundary::collapse(&[1.0,1.0],&[1.00390625,0.00390625],2,1,0).to_bits(),1.0078125f32.to_bits());
    assert_eq!(boundary::collapse(&[1.0],&[1.00390625],1,1,0).to_bits(),1.0f32.to_bits());
    assert_eq!(boundary::collapse(&[1.0],&[1.01171875],1,1,0).to_bits(),1.015625f32.to_bits());
}
#[test]
#[allow(clippy::excessive_precision)] // literal constants stay as written.
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
#[allow(clippy::excessive_precision)] // literal constants stay as written.
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
fn grouped_square_loads_keep_the_fixed_rms_tree_at_width_and_batch_tails() {
    use spark_gpu_probe::{mhc_params::Shape,mhc_reference};
    let mut normalized_cases=0;let mut chunk_cases=0;let mut changed=0;
    for streams in [1u32,2,4,7] {for hidden in [1u32,2,7,8,9,17,128,515,4096] {
        for batch in [1u32,2,8,127,128,129,255,256,257] {for token in [0,batch-1] {
            let shape=Shape {rows:batch,streams,hidden,iterations:4,rms_eps:1e-5,eps:1e-5};shape.validate().unwrap();
            let input=(0..shape.wide()).map(|column|if column%13==0 {-0.0} else {((column+token as usize*7) as i32%31-15) as f32*0.037}).collect::<Vec<_>>();
            let products=input.iter().map(|value|value*value).collect::<Vec<_>>();
            let mut lanes=[0.0f32;32];
            for (lane,slot) in lanes.iter_mut().enumerate() {for strip in products.chunks(8).skip(lane).step_by(32) {for &value in strip {*slot+=value;}}}
            for distance in [16,8,4,2,1] {for lane in 0..distance {lanes[lane]+=lanes[lane+distance];}}
            let squares=lanes[0];
            assert_eq!(spark_gpu_probe::reduction::strip_fold(input.len(),|column|input[column]*input[column]).to_bits(),squares.to_bits());
            changed+=usize::from(products.iter().fold(0.0f32,|sum,value|sum+value).to_bits()!=squares.to_bits());
            for chunk in [1usize,8,16,128,256,1024,1025] {
                let mut streamed=spark_gpu_probe::reduction::StripFold::new();
                for tile in products.chunks(chunk) {for &value in tile {streamed+=value;}}
                assert_eq!(streamed.finish().to_bits(),squares.to_bits(),"streams={streams} hidden={hidden} batch={batch} chunk={chunk}");chunk_cases+=1;
            }
            let inverse=1.0/(squares/shape.wide() as f32+shape.rms_eps).sqrt();
            let expected=input.iter().map(|&value|(value*inverse).to_bits()).collect::<Vec<_>>();let mut actual=vec![f32::NAN;input.len()];
            mhc_reference::normalize(shape,&input,&mut actual);
            assert_eq!(actual.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),expected,"streams={streams} hidden={hidden} batch={batch}");normalized_cases+=1;
        }}
    }}
    assert!(changed>0);
    println!("ORDER_TREE RMS normalized_cases={normalized_cases} chunk_cases={chunk_cases} changed_from_serial={changed} batches=1,2,8,127,128,129,255,256,257 chunks=1,8,16,128,256,1024,1025");
}

#[test]
fn chain_shadow_fields_follow_the_original_boundary_abi() {
    use spark_gpu_probe::mhc_params::{AddLatency,Boundary};use std::mem::{offset_of,size_of};
    // The first 136 bytes keep their old offsets; the CHAIN-SHADOW fields follow them.
    assert_eq!((offset_of!(Boundary,norm_epsilon),offset_of!(Boundary,norm_bf16)),(128,132));
    assert_eq!((offset_of!(Boundary,prefetch),offset_of!(Boundary,prefetch_bytes),offset_of!(Boundary,prefetch_policy),offset_of!(Boundary,prefetch_reserved)),(136,144,152,156));
    assert_eq!((size_of::<AddLatency>(),offset_of!(AddLatency,start),offset_of!(AddLatency,step),offset_of!(AddLatency,adds)),(24,8,12,16));
}

#[test]
fn chain_shadow_tail_issues_each_line_once_inside_the_span() {
    use spark_gpu_probe::mhc_params::{SHADOW_LINE,SHADOW_MAX_BLOCKS,shadow_lines};
    // Mirrors chain_shadow::shadow_tail: tail block b of `blocks`, thread t of `threads`.
    let base=0x7f00_0000_0040u64;
    for bytes in [1u64,127,128,129,4096,1_000_003,8_650_752] {for blocks in [1u32,3,24,SHADOW_MAX_BLOCKS] {for threads in [256u64,768] {
        let lines=shadow_lines(bytes);let mut seen=vec![0u8;lines as usize];let stride=u64::from(blocks)*threads;
        for b in 0..u64::from(blocks) {for t in 0..threads {
            let mut line=b*threads+t;
            while line<lines {let address=base+line*SHADOW_LINE;assert!(address<base+bytes);seen[line as usize]+=1;line+=stride;}
        }}
        assert!(seen.iter().all(|&count|count==1),"bytes={bytes} blocks={blocks} threads={threads}");
        assert!(base+(lines-1)*SHADOW_LINE+SHADOW_LINE>=base+bytes);
    }}}
}

#[test]
fn chain_add_probe_replay_matches_the_measured_device_bits() {
    use spark_gpu_probe::mhc_params::{SHADOW_DECODE_BLOCKS,SHADOW_DECODE_BYTES,SHADOW_MAX_BLOCKS,serial_add_bits};
    // Device results printed by `kernel-bench addlat` on one Spark, screen de9aa38ef (session-shadow.log).
    assert_eq!(serial_add_bits(1.0,1.0/1024.0,16384),0x41880000);
    assert_eq!(serial_add_bits(1.0,1.0/1024.0,262144),0x43808000);
    assert_eq!(serial_add_bits(1.0,1.0/1024.0,0),1.0f32.to_bits());
    // The serial order matters: 2^24 + 1 + 1 stays 2^24 one add at a time.
    assert_eq!(serial_add_bits(16_777_216.0,1.0,2),16_777_216.0f32.to_bits());
    assert!(SHADOW_DECODE_BYTES>0 && (1..=SHADOW_MAX_BLOCKS).contains(&SHADOW_DECODE_BLOCKS));
}

#[test]
fn chain_add_probe_counts_cannot_wrap_at_any_u32() {
    use spark_gpu_probe::mhc_params::{PROBE_MAX_ADDS,probe_chunks};
    // The review's case: 4,294,967,280 and above made `done + 16` wrap in the old loop.
    for adds in [0u32,1,15,16,17,16384,262144,PROBE_MAX_ADDS,4_294_967_279,4_294_967_280,u32::MAX-1,u32::MAX] {
        let (groups,tail)=probe_chunks(adds);
        assert!(tail<16);assert_eq!(u64::from(groups)*16+u64::from(tail),u64::from(adds),"adds={adds}");
        // The kernel counter runs 0..groups, then the tail counts down: neither exceeds its bound.
        assert!(groups<=u32::MAX/16);
    }
    // The bench binder bound stays far below any wrap and covers the bench's longest chain.
    assert!(PROBE_MAX_ADDS>=262144 && u64::from(PROBE_MAX_ADDS)<u64::from(u32::MAX));
}
