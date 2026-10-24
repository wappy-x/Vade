use spark_gpu_probe::next_params::{MxLayer,MxProjection};
#[test]
fn native_dispatch() {
    for m in 1..=16 { for (n,k) in [(48,2560),(640,2560),(10240,2560),(2560,6144),(2560,640)] {
        let mut l=MxLayer::default(); let p=MxProjection {weights:16,scales:32,input:48,output:64,rows:n,cols:k,tokens:m,..Default::default()};
        l.push(p).unwrap(); assert_eq!(l.ends[1],n.div_ceil(4)*m.div_ceil(p.tile()));
        assert!(l.push(MxProjection {cols:k+1,..p}).is_err());
    } }
    assert_eq!((std::mem::size_of::<MxProjection>(),std::mem::size_of::<MxLayer>()),(56,552));
}
// Real text MXFP8 shape families; cancellation exposes loss of the FP32
// residual before accumulation. No checkpoint or GPU is needed for this gate.
#[test]
fn projection_f32_residual_real_shapes() {
    use spark_gpu_probe::bf16::rounded;
    let shapes = [(48,2560),(10240,2560),(6144,2560),(2560,6144),
        (12288,2560),(512,2560),(640,2560),(2560,640)];
    for (n,k) in shapes {
        let p = MxProjection { rows:n, cols:k, tokens:1, ..Default::default() };
        let mut first = Vec::new();
        for repeat in 0..3 {
            let (mut peak,mut err,mut plain_err,mut old_err) = (0f64,0f64,0f64,0f64);
            for r in 0..n {
                let (mut lanes,mut old,mut hi,mut lo) = ([0f32;32],[0f32;32],[0f32;32],[0f32;32]);
                let mut cpu = 0f64;
                for c in 0..k {
                    // E4M3 mantissa and a per-32 UE8M0 power-of-two scale.
                    let w = (1.0+(r%8) as f32/8.0)*[0.25,0.5,1.0,2.0][(c/32%4) as usize]
                        * if c%2==0 {1.0} else {-1.0};
                    let x = if c%2==0 {1.0+1.0/512.0} else {1.0};
                    let lane = (c%256/8) as usize;
                    let h = rounded(x); let l = rounded(x-h);
                    lanes[lane] = w.mul_add(p.f32_input(x.to_bits()),lanes[lane]);
                    old[lane] = w.mul_add(h,old[lane]);
                    hi[lane] = w.mul_add(h,hi[lane]); lo[lane] = w.mul_add(l,lo[lane]);
                    cpu += w as f64*x as f64;
                }
                for d in [16,8,4,2,1] { for lane in 0..d {
                    lanes[lane] += lanes[lane+d]; old[lane] += old[lane+d];
                    hi[lane] += hi[lane+d]; lo[lane] += lo[lane+d];
                } }
                peak = peak.max(cpu.abs()); err = err.max((lanes[0] as f64-cpu).abs());
                plain_err = plain_err.max(((hi[0]+lo[0]) as f64-cpu).abs());
                old_err = old_err.max((old[0] as f64-cpu).abs());
                if repeat==0 { first.push(lanes[0].to_bits()); }
                else { assert_eq!(first[r as usize],lanes[0].to_bits(),"repeat {repeat}"); }
            }
            assert!(old_err>1e-5+peak/128.0,"fixture must reject old input rounding");
            assert!(err<=plain_err,"[{n},{k}] PROJ={err} > PLAIN={plain_err}");
            assert!(err<=1e-5+peak/128.0,"[{n},{k}] normwise regression");
        }
        let rounded_p = MxProjection { round:1, ..p };
        let x = 1.0f32+1.0/512.0;
        assert_eq!(rounded_p.f32_input(x.to_bits()),rounded(x),"GDN rounding retained");
        eprintln!("[{n},{k}] three repeats identical; PROJ error <= PLAIN; old path rejected");
    }
}
#[cfg(target_os="linux")]
mod gpu {
    use super::*;
    use spark_gpu_probe::driver::{Backend,DeviceVector,with_backend};
    fn root() -> std::path::PathBuf {std::env::var_os("VADE_NEXT_FIXTURES").expect("set VADE_NEXT_FIXTURES to unchanged prototype exports").into()}
    fn upload<'a>(b:&Backend<'a>,p:&std::path::Path)->Result<DeviceVector<'a>,String>{let bytes=std::fs::read(p).map_err(|e|e.to_string())?;let mut v=b.vector(bytes.len().div_ceil(4))?;b.write_vector_bytes(&mut v,&bytes)?;Ok(v)}
    fn bytes(b:&Backend<'_>,v:&DeviceVector<'_>)->Result<Vec<u8>,String>{Ok(b.read_vector(v)?.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect())}
    fn equal(b:&Backend<'_>,v:&DeviceVector<'_>,p:&std::path::Path)->Result<(),String>{assert_eq!(bytes(b,v)?,std::fs::read(p).map_err(|e|e.to_string())?,"{}",p.display());Ok(())}
    #[test] #[ignore = "skipped: no GPU; requires GB10 and unchanged CUDA prototype fixtures"]
    fn mxfp8_prototype_bytes()->Result<(),String>{with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b|{
        let root=root(); let text=std::fs::read_to_string(root.join("mx.txt")).map_err(|e|e.to_string())?; let lines:Vec<_>=text.lines().collect(); assert_eq!(lines.len(),16*9);
        for group in lines.chunks_exact(9) {let mut cases=Vec::new(); for line in group {
            let t:Vec<_>=line.split_whitespace().collect();let p=root.join(t[0]);let z:Vec<u32>=t[1..].iter().map(|s|s.parse().unwrap()).collect();let (n,k,m,round)=(z[0],z[1],z[2],z[3]);
            let w=upload(b,&p.join("w"))?;let s=upload(b,&p.join("s"))?;let x=upload(b,&p.join("x"))?;let xf=upload(b,&p.join("xf"))?;let y=b.vector((n*m) as usize)?; cases.push((p,n,k,m,round,w,s,x,xf,y));
        }
        for input_bf16 in [1,0] { for subset in [0..5,5..9] {let mut l=MxLayer::default();for c in &cases[subset.clone()] {l.push(MxProjection{weights:c.5.device_address(),scales:c.6.device_address(),input:if input_bf16==1{c.7.device_address()}else{c.8.device_address()},output:c.9.device_address(),rows:c.1,cols:c.2,tokens:c.3,input_bf16,round:c.4}).map_err(str::to_string)?;}
            for _ in 0..2 {unsafe{b.launch_flash_shared(l.projections[0].kernel(),l.ends[l.count as usize],128,0,&mut[(&mut l as *mut MxLayer).cast()])?;} for c in &cases[subset.clone()] {equal(b,&c.9,&c.0.join("y"))?;}}
        } }
        for c in &cases {let w=std::fs::read(c.0.join("w")).map_err(|e|e.to_string())?;let scales=std::fs::read(c.0.join("s")).map_err(|e|e.to_string())?;let x=b.read_vector(&c.8)?;let y=b.read_vector(&c.9)?;let (mut max_ref,mut max_err)=(0.0f64,0.0f64);
            for r in [0,c.1 as usize/2,c.1 as usize-1] {for t in 0..c.3 as usize {let (mut dot,mut bound)=(0.0f64,0.0f64);for k in 0..c.2 as usize {let at=r*c.2 as usize+k;let code=w[at];let mag=code&127;let v=if mag<8 {mag as f64/512.0}else{(1.0+(mag&7) as f64/8.0)*2f64.powi((mag>>3) as i32-7)};let v=if code&128==0{v}else{-v};let term=v*2f64.powi(scales[at/32] as i32-127)*x[t*c.2 as usize+k] as f64;dot+=term;bound+=term.abs();}max_ref=max_ref.max(dot.abs());max_err=max_err.max((y[t*c.1 as usize+r] as f64-dot).abs());let _=bound;}} assert!(max_err<=1e-5+max_ref/128.0,"sampled normwise BF16 CPU tolerance: error={max_err} scale={max_ref}");
        }
        } Ok(())})}
}
