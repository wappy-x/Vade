use crate::{expert_ops::{e2m1,e4m3,nearest_e2m1,round},mhc_params::Shape,mhc_reference,reduction::strip_fold};

fn oracle(terms:&[f32])->f32 {
    let mut level=(0..32).map(|lane|terms.chunks(8).skip(lane).step_by(32).flatten()
        .fold(0.0f32,|sum,value|sum+value)).collect::<Vec<_>>();
    while level.len()>1 {let half=level.len()/2;level=(0..half).map(|lane|level[lane]+level[lane+half]).collect();}
    level[0]
}

fn warp_model(terms:&[f32])->f32 {
    let mut lanes=[0.0f32;32];
    for (lane,slot) in lanes.iter_mut().enumerate() {
        let mut first=lane*8;
        while first<terms.len() {
            for offset in 0..8 {if first+offset<terms.len() {*slot+=terms[first+offset];}}
            first+=256;
        }
    }
    crate::reduction::warp_fold_join!(lanes,|before:[f32;32],distance:u32| {
        core::array::from_fn::<_,32,_>(|lane|before[if lane+(distance as usize)<32 {lane+distance as usize} else {lane}])
    },|before:[f32;32],other:[f32;32]|core::array::from_fn(|lane|before[lane]+other[lane]));
    lanes[0]
}

#[test]
fn warp_fold_expansion_matches_strip_fold() {
    let mut cases=0;
    for columns in [0usize,1,7,8,9,31,32,33,127,128,129,255,256,257,511,512,513,4095,4096,4097,16384] {
        for format in 0..3 {for token in 0..8 {
            let values=terms(format,token,columns);
            assert_eq!(warp_model(&values).to_bits(),strip_fold(columns,|column|values[column]).to_bits(),"columns={columns} format={format} token={token}");
            cases+=1;
        }}
        for active_lane in 0..32 {
            let values=(0..columns).map(|column|if column/8%32==active_lane {
                [16_777_216.0,1.0,-16_777_216.0,-0.0,f32::from_bits(1)][column%5]
            } else {0.0}).collect::<Vec<_>>();
            assert_eq!(warp_model(&values).to_bits(),strip_fold(columns,|column|values[column]).to_bits(),"columns={columns} active_lane={active_lane}");
            cases+=1;
        }
    }
    println!("ORDER_TREE warp_fold_shared_expansion_cases={cases} shuffle=all_lanes_synchronous_self_on_overflow lane_zero=bit_equal");
}

fn terms(format:usize,token:usize,columns:usize)->Vec<f32> {
    (0..columns).map(|column| {
        let input=match column%13 {0=>-0.0,1=>16_777_216.0,2=>-16_777_216.0,3=>f32::from_bits(1),_=>((column*7+token*13)%31) as f32/17.0-1.0};
        let weight=((column*37+token*11)%126) as u8|if column%5==0 {128} else {0};
        match format {
            0=>input*((column*17%101) as f32/64.0-0.5),
            1=>(e4m3(weight)*(0.03125*(1+column/128%31) as f32))*round(input),
            _=>{let fold=(e4m3((8+column/16%32) as u8)*0.375)*(0.3*0.125);
                (e2m1(weight&15)*e2m1(nearest_e2m1(input)))*fold},
        }
    }).collect()
}

#[test]
fn strip_fold_matches_independent_tree_for_all_formats_and_batches() {
    let mut cases=0;let mut changed=[0usize;3];
    for (format,count) in changed.iter_mut().enumerate() {for columns in [0,1,2,7,8,9,17,255,256,257,4095,4096,4097,16384] {
        let singles=(0..8).map(|token|{let values=terms(format,token,columns);strip_fold(columns,|column|values[column]).to_bits()}).collect::<Vec<_>>();
        for batch in [1,2,8] {for (token,&single) in singles.iter().enumerate().take(batch) {
            let values=terms(format,token,columns);let actual=strip_fold(columns,|column|values[column]);
            assert_eq!(actual.to_bits(),oracle(&values).to_bits(),"format={format} columns={columns} token={token}");
            assert_eq!(actual.to_bits(),warp_model(&values).to_bits());
            assert_eq!(actual.to_bits(),single);
            let serial=values.iter().fold(0.0f32,|sum,value|sum+value);
            *count+=usize::from(serial.to_bits()!=actual.to_bits());cases+=1;
        }}
    }}
    assert!(changed.iter().all(|&count|count>0));
    println!("ORDER_TREE formats=FP32,FP8-block,NVFP4 batches=1,2,8 oracle_cases={cases} changed_from_serial={changed:?}");
}

#[test]
fn strip_fold_keeps_product_rounding_signed_zero_and_subnormals() {
    let above=f32::from_bits(0x3f800001);let below=f32::from_bits(0x3f7ffffe);
    let product=above*below;
    assert_eq!(product.to_bits(),1.0f32.to_bits());
    assert_eq!(strip_fold(2,|column|if column==0 {-1.0} else {above*below}).to_bits(),0);
    assert_eq!(strip_fold(257,|_|-0.0).to_bits(),0);
    assert_eq!(strip_fold(257,|_|f32::from_bits(1)).to_bits(),257);
    let mut terms=vec![0.0;257];terms[0]=16_777_216.0;terms[8]=1.0;terms[256]= -16_777_216.0;
    assert_eq!(strip_fold(terms.len(),|column|terms[column]).to_bits(),1.0f32.to_bits());
    assert_eq!(terms.iter().fold(0.0f32,|sum,value|sum+value).to_bits(),0);
    println!("ORDER_TREE separate_product_rounding=pass signed_zero=pass subnormals=pass cancellation=pass");
}

#[test]
fn strip_fold_dependency_count_is_not_a_timing_measurement() {
    let columns=16384;let mut lanes=[0usize;32];
    for column in 0..columns {lanes[column/8%32]+=1;}
    let tree=lanes.into_iter().max().unwrap()+5;
    assert_eq!(tree,517);
    println!("ORDER_TREE dependency_model columns={columns} serial_add_depth={columns} tree_add_depth={tree} ratio={:.6} measured_ms=none",columns as f64/tree as f64);
}

#[test]
fn mhc_coefficients_match_tree_across_split_fallback_and_prompt_batches() {
    let mut cases=0;
    for streams in [1,3,4,7] {for hidden in [1,7,65,1025,4096] {
        let single=Shape {rows:1,streams,hidden,iterations:4,rms_eps:1e-6,eps:1e-6};
        let wide=single.wide();let count=single.coefficients();
        let weights=(0..count*wide).map(|index|((index*37%101) as f32-50.0)/31.0).collect::<Vec<_>>();
        let base=(0..count).map(|index|(index as f32-3.0)/71.0).collect::<Vec<_>>();let scale=[0.3,0.125,0.0625];
        for token in 0..8 {
            let input=(0..wide).map(|column|round(((column*13+token*7)%59) as f32/19.0-1.0)).collect::<Vec<_>>();
            let mut normalized=vec![0.0;wide];let mut expected=vec![0.0;count];
            mhc_reference::prepare(single,&input,&weights,&base,&scale,&mut normalized,&mut expected);
            for batch in [1,2,8,9,17] {
                let shape=Shape {rows:batch,..single};let inverse=mhc_reference::inverse(shape,&input);
                let mut raw=vec![0.0;count];
                let warps=shape.prepare_threads() as usize/32;
                for warp in 0..warps {for coefficient in (warp..count).step_by(warps) {
                    let products=(0..wide).map(|column|weights[coefficient*wide+column]*(input[column]*inverse)).collect::<Vec<_>>();
                    let group=if coefficient<streams as usize {0} else if coefficient<2*streams as usize {1} else {2};
                    raw[coefficient]=oracle(&products)*scale[group]+base[coefficient];
                    assert_eq!(raw[coefficient].to_bits(),mhc_reference::project(shape,coefficient,&weights,&base,&scale,&normalized).to_bits());
                }}
                mhc_reference::activate(shape,&mut raw);
                assert_eq!(raw.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),expected.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
                cases+=1;
            }
        }
    }}
    println!("ORDER_TREE mhc_batches=1,2,8,9,17 complete_coefficient_cases={cases}");
}
