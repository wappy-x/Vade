use spark_engine::arms;
#[allow(dead_code, clippy::manual_is_multiple_of)] #[path = "../../gpu/src/projection_params.rs"] mod params; // Shared with the NVPTX crate.
#[allow(dead_code)] #[path = "../src/hybrid/linear.rs"] mod linear;
#[cfg(not(target_os = "linux"))] #[allow(dead_code)] #[path = "../src/hybrid/gdn.rs"] mod gdn;
#[cfg(target_os = "linux")] use spark_engine::hybrid::decode::gdn;
#[test]
fn fixed_projection_geometry() {
    for (h,a,q,v,i) in [(2048,4096,8192,32,512),(3072,8192,16384,64,1024)] { for (n, k, fp8) in [(q,h,true),(512,h,true),(h,a,true),(4*h,h,true),(a,h,true),(i,h,true),(h,i,true),(i/2,h,true),(h,i/2,true),(h,a/2,true),(v,h,false),(256,h,false),(1,h,false),(248320,h,false),(124160,h,false),(h,2*h,false)] {
        let p = params::FixedLinear::new(n,k,fp8).unwrap(); let order = p.k_ranges(); let tile = p.tile();
        assert_eq!(order.first().unwrap().start,0); assert_eq!(order.last().unwrap().end,k as u32); assert!(order.windows(2).all(|r| r[0].end == r[1].start)); assert_eq!(order.len(),if fp8 { k/128 } else { 4 }); if fp8 { assert_eq!(tile,[128,128,512,66560]); }
        for m in [1usize,8,16,32,256,257,768,1024] { assert_eq!(p.elements(m).unwrap(),(m*k,m*n)); assert_eq!(p.k_ranges(),order); assert_eq!(p.tile(),tile); assert!(m.div_ceil(tile[0] as usize)*tile[0] as usize-m < tile[0] as usize); if fp8 { let (s,w) = p.scratch(m).unwrap(); assert!(s >= m*k && s % 256 == 0 && w*4 == s+m*k/128*4); for r in 0..m { for c in (0..k).step_by(128) { assert_eq!((r*k+c)/128-r*(k/128),c/128); } } } }
        if !fp8 { assert_eq!(tile[2] == 128,spark_gpu_probe::kernel_params::flash_bf16_decode_split(n,k)); }
        for m in [1usize,7,8,9,255,256] { let launch=p.tile_reusing_weights(m,true);
            if !fp8 && m>8 && n<=512 && k<=4096 && k%64==0 { assert_eq!(launch,[256,8,256,(16*k) as u32]); assert_eq!(m.div_ceil(launch[0] as usize),1); }
            else { assert_eq!(launch,tile); } assert_eq!(p.tile_reusing_weights(m,false),tile);
        }
    } }
    for (n,k,f) in [(0,128,true),(128,0,true),(129,128,true),(128,129,true),(1,17,false),(usize::MAX,128,true),(u32::MAX as usize,4294967280,false)] { assert!(params::FixedLinear::new(n,k,f).is_err()); }
    let p = params::FixedLinear::new(128,128,true).unwrap(); assert!(p.elements(0).is_err() && p.elements(1025).is_err()); assert!(params::FixedLinear::new(1,16,false).unwrap().scratch(1).is_err());
    assert_ne!(linear::OutputBoundary::Fp32,linear::OutputBoundary::Bf16); assert!(params::FixedLinear::disjoint(256,128,384,128).is_ok()); assert!(params::FixedLinear::disjoint(256,129,384,128).is_err() && params::FixedLinear::disjoint(u64::MAX,1,256,1).is_err());
    for h in [2048,3072] { let norm=linear::NormBinding::new(h,1e-6).unwrap(); for rows in [1,2,8,16,32,256,32768] { assert_eq!(norm.elements(rows).unwrap(),h*rows); assert_eq!((linear::NormBinding::ORDER,linear::NormBinding::THREADS),(2,128)); } assert!(norm.elements(0).is_err()); }
    for (h,e) in [(0,1e-6),(5121,1e-6),(2048,0.0),(3072,f32::NAN)] { assert!(linear::NormBinding::new(h,e).is_err()); }
    use engine_model::shapes::{Role,Place,Part,Store}; for (place,part,boundary) in [(Place::Layer(0),Part::LinearHeads,linear::OutputBoundary::Bf16),(Place::Draft{layer:Some(0)},Part::LinearHeads,linear::OutputBoundary::Bf16),(Place::Draft{layer:None},Part::Whole,linear::OutputBoundary::Bf16),(Place::Layer(0),Part::Whole,linear::OutputBoundary::Fp32),(Place::Model,Part::Head,linear::OutputBoundary::Fp32)] { assert_eq!(linear::bf16_boundary(Role{place,part,store:Store::Weight}).unwrap(),boundary); }
    for (place,part,store) in [(Place::Unused,Part::Head,Store::Weight),(Place::Draft{layer:None},Part::Whole,Store::BlockScale),(Place::Layer(0),Part::LinearHeads,Store::BlockScale)] { assert!(linear::bf16_boundary(Role{place,part,store}).is_err()); }
}
#[test]
fn gdn_plan_binding() { use engine_model::{plan::ModelPlan,safetensors::Dtype,shapes::{Mixer,Part,Place,Role,Store,TensorEntry}}; for (bytes,bits) in [(include_bytes!("../../model/tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json").as_slice(),1),(include_bytes!("../../model/tests/fixtures/qwen/Qwen3.5-122B-A10B-FP8.config.json").as_slice(),11)] { let mut p=ModelPlan::from_config(bytes).unwrap(); let Mixer::Linear(l)=p.shape.layers[0].mixer else { panic!() }; for (name,part,n,flag) in [("A_log",Part::LinearHeads,l.value_heads,2),("dt_bias",Part::LinearHeads,l.value_heads,4),("norm.weight",Part::LinearNorm,128,8)] { p.shape.tensors.push(TensorEntry{name:format!("linear_attn.{name}"),dtype:if bits&flag==0 {Dtype::Bf16} else {Dtype::F32},shape:vec![n],role:Role{place:Place::Layer(0),part,store:Store::Weight}}); } let b=gdn::StepBinding::whole(&p,0).unwrap(); assert_eq!(b.geometry(),[16,l.value_heads as u32,0,0,bits]); assert_eq!(b.projection_rows(),[((32+l.value_heads)*128) as usize,(l.value_heads*128) as usize,l.value_heads as usize,l.value_heads as usize]); assert!(b.execution_ready().is_ok()); p.shape.tensors[0].role.part=Part::Whole; assert!(gdn::StepBinding::whole(&p,0).is_err()); } }

// Portable operand/FP32 schedule check; this does not execute CUDA MMA instructions.
#[test]
fn bf16_head_mma_schedule_bitwise() {
    use spark_gpu_probe::{bf16,kernel_params};
    let split = |x: f32| { let hi=bf16::rounded(x); [hi,bf16::rounded(x-hi)] };
    let value = |i: usize| { let b=(i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
        f32::from_bits((b & 0x807fffff) | (0x3e800000 + ((b >> 24 & 3) << 23))) };
    let packed_at = |t: usize,c: usize,k: usize| ((t/16*(k/16)+c/16)*32+t%8*4+c%8/2)*8+t%16/8*2+c%16/8*4+c%2;
    for (n,k) in [(248320usize,2048usize),(2048,4096)] {
        let ranges=params::FixedLinear::new(n,k,false).unwrap().k_ranges();
        for m in [1usize,7,8,9,16,17,31,32,33,63,64] {
            let mut packed=vec![[0.0;2];m.div_ceil(16)*16*k];
            for t in 0..m { for c in 0..k { packed[packed_at(t,c,k)]=split(value(t*k+c)); } }
            for t in 0..m { for r in (0..64).chain(n-64..n) {
                let mut narrow=[0.0;4];
                for (part,range) in ranges.iter().enumerate() { let mut acc=[0.0f32;2];
                    for c in range.start as usize..range.end as usize {
                        let (rt,ct)=(r/8,c/16);
                        let at=((rt/8*(k/64)+ct/4)*32+rt%8*4+ct%4)*128+(r%8*4+c%8/2)*4+c%2+c%16/8*2;
                        assert_eq!(at,kernel_params::flash_bf16_element(r,c,k,n));
                        let a=split(value(t*k+c)); let b=bf16::rounded(value(at));
                        for p in 0..2 { acc[p]=a[p].mul_add(b,acc[p]); }
                    } narrow[part]=acc[0]+acc[1];
                }
                let (mut acc,mut wide,mut part)=([0.0f32;2],[0.0;4],0);
                for tile in (0..k/16).step_by(4) { for z in 0..4 { for c in (tile+z)*16..(tile+z+1)*16 {
                    let a=packed[packed_at(t,c,k)]; let b=bf16::rounded(value(kernel_params::flash_bf16_element(r,c,k,n)));
                    assert_eq!(a.map(f32::to_bits),split(value(t*k+c)).map(f32::to_bits));
                    for p in 0..2 { acc[p]=a[p].mul_add(b,acc[p]); }
                } if (tile+z+1)*16==ranges[part].end as usize { wide[part]=acc[0]+acc[1]; acc=[0.0;2]; part+=1; } } }
                assert_eq!(((narrow[0]+narrow[1])+(narrow[2]+narrow[3])).to_bits(),
                    ((wide[0]+wide[1])+(wide[2]+wide[3])).to_bits(),"N={n} K={k} M={m} t={t} r={r}");
            } }
            for t in m..m.div_ceil(16)*16 { for c in 0..k { assert_eq!(packed[packed_at(t,c,k)],[0.0;2]); } }
        }
    }
}
#[cfg(target_os = "linux")]
#[test]
#[ignore = "skipped: no GPU; single GB10 qualification: executes both CUDA entries"]
fn bf16_head_cuda_entries_bitwise() -> Result<(), String> {
    use spark_gpu_probe::{driver::with_backend,kernel_params::{Bf16Rows,FlashBf16Input,FLASH_BF16_MMA_SHARED,TERMS}};
    with_backend(0,spark_gpu_probe::layer_ptx()?,2*1024*1024*1024,|b| {
        for (n,k) in [(248320usize,2048usize),(2048,4096)] {
            let weight=b.upload_packed_bf16(n,k,1024*1024,|offset,bytes| {
                for (i,pair) in bytes.as_chunks_mut::<2>().0.iter_mut().enumerate() { let j=offset/2+i;
                    let v=(0x3d00+((j*13+j/k*7)%512) as u16) | if j%3==0 {0x8000} else {0}; pair.copy_from_slice(&v.to_le_bytes()); } Ok(()) })?;
            for m in [1usize,7,8,9,16,17,31,32,33,63,64] {
                let input=b.upload_vector(&(0..m*k).map(|i| ((i*17+i/k*11)%257) as f32/129.003-0.99).collect::<Vec<_>>())?;
                let plane=m.div_ceil(16)*k/16*128;
                let scratch=b.vector(plane*TERMS)?; let mut output=b.vector((m+1)*n)?;
                let poison=vec![f32::from_bits(0x7fc00123);(m+1)*n];
                b.write_vector(&mut output,&poison)?;
                let mut p=Bf16Rows { weight:weight.device_address(),output:output.device_address(),rows:n as u32,stride:n as u32,
                    weight2:0,output2:0,rows2:0,stride2:0,input:input.device_address(),cols:k as u32,tokens:m as u32,
                    input_bf16:0,hc_silu:0,input_plane_words:0 };
                // Owned disjoint allocations remain live through synchronized readback.
                unsafe { b.launch_flash_rows("flash_bf16_mma1_rows",n.div_ceil(16) as u32,m.div_ceil(8) as u32,64,0,
                    &mut [(&mut p as *mut Bf16Rows).cast()])?; }
                let expected:Vec<_>=b.read_vector_values(&output)?.into_iter().map(f32::to_bits).collect();
                assert!(expected[..m*n].iter().all(|&x|f32::from_bits(x).is_finite()));
                assert!(expected[..m*n].iter().any(|&x|f32::from_bits(x)!=0.0));
                b.write_vector(&mut output,&poison)?;
                let mut pack=FlashBf16Input {input:input.device_address(),output:scratch.device_address(),tokens:m as u32,cols:k as u32,hc_silu:0,lanes:0};
                p.input=scratch.device_address(); p.input_bf16=1; p.input_plane_words=plane as u64;
                unsafe { b.launch_flash_shared("flash_bf16_input",plane.div_ceil(4).div_ceil(128) as u32,128,0,
                    &mut [(&mut pack as *mut FlashBf16Input).cast()])?;
                    b.launch_flash_shared("flash_bf16_mma64",(n.div_ceil(64)*m.div_ceil(64)) as u32,256,FLASH_BF16_MMA_SHARED as u32,
                    &mut [(&mut p as *mut Bf16Rows).cast()])?; }
                assert_eq!(expected.iter().zip(b.read_vector_values(&output)?).position(|(&a,b)|a!=b.to_bits()),None,"N={n} K={k} M={m}: first differing word");
                assert!(expected[m*n..].iter().all(|&x|x==0x7fc00123),"masked output guard");
            }
        } Ok(())
    })
}

#[allow(dead_code)] #[path = "../src/hybrid/schedule.rs"] mod schedule;
#[test]
fn paired_router_gate_and_distributed_boundary_compose() {
    // Pairing applies independently of transport: distributed decode must
    // retain both optimizations, while prompt rows retain both fallbacks.
    let verify_rows = 80;
    #[cfg(target_os = "linux")]
    assert_eq!(verify_rows, spark_engine::hybrid::decode::MAX_VERIFY_ROWS);
    for (experts, hidden) in [(256, 2048), (256, 3072), (512, 3072)] {
        for distributed in [false, true] {
            for (rows, pair, fuse) in [(1,true,true),(4,true,true),(8,true,true),(9,false,true),
                (16,false,true),(80,false,true),(81,false,false),(256,false,false),(1024,false,false)] {
                assert_eq!((linear::router_gate_pair_rows(experts,1,hidden,hidden,rows),
                    schedule::fused_router_boundary(distributed,rows,verify_rows)), (pair,distributed && fuse));
            }
        }
    }
    // Unsupported paired geometry must not disable distributed fusion.
    assert!(!linear::router_gate_pair_rows(1023,1,2048,2048,1));
    assert!(!linear::router_gate_pair_rows(256,1,2048,3072,1));
    assert!(schedule::fused_router_boundary(true,1,verify_rows));
}
