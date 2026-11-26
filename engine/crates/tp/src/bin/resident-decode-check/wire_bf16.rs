//! Byte and wrong-control screen of the lossless wire representation. No weights.
use std::{path::Path,time::Instant};
use spark_gpu_probe::{driver::Backend,prompt_ops::{self,Shape,Op}};
use spark_tp::Result;
#[path="wire_bf16/prompt.rs"] pub(super) mod prompt;
fn run_case(b:&Backend<'_>,tokens:usize,ranks:usize,width:usize,transpose:bool,queued:bool)->Result<usize> {
    let words=tokens*ranks*width;
    let input=(0..words).map(|i| {
        let mut bits=((i as u32).wrapping_mul(17)%65536)<<16;
        if bits&0x7f800000==0x7f800000 {bits^=0x00800000;}f32::from_bits(bits)
    }).collect::<Vec<_>>();
    let source=b.upload_vector(&input)?.into_resident_shared()?;let mapped=b.mapped_vector(words/2+8)?;
    let guards=b.upload_vector(&vec![f32::from_bits(0x5a5aa5a5);words/2+8])?;
    b.copy_prefix(&guards,&mut {mapped.resident_range(0..mapped.len())?},mapped.len())?;b.synchronize()?;
    let queue=if queued {let q=b.resident_queue()?;b.enable_resident_launches()?;Some(q)} else {None};
    let pack=Shape {cols:words as u32,..Shape::new(Op::WirePackBf16,1).ok_or("pack shape")?};
    let packet=b.prompt_operation_output(pack,[Some(&source),None,None,None,None],None,Some(mapped.resident_range(4..4+words/2)?))?;
    let (unpack,old)=if transpose {(Shape {rows:(ranks*width) as u32,cols:tokens as u32,first:1,..Shape::new(Op::WireUnpackBf16,1).ok_or("unpack shape")?},Shape {rows:(ranks*width) as u32,cols:tokens as u32,..Shape::new(Op::Transpose,1).ok_or("old shape")?})}
        else {(Shape {rows:ranks as u32,cols:width as u32,..Shape::new(Op::WireUnpackBf16,tokens).ok_or("unpack shape")?},Shape::rank_rows(tokens,width,ranks,true).ok_or("old shape")?)};
    let output=b.prompt_operation(unpack,[Some(&packet),None,None,None,None],None)?;
    drop(queue);let got=b.read_vector_values(&output)?;let expected=prompt_ops::reference(old,[&input,&[],&[],&[],&[]],&[])?;
    if got.iter().zip(&expected).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err("wire BF16 byte mismatch".into());}
    let packed=b.read_vector_values(&packet)?;let expected_packet=prompt_ops::reference(pack,[&input,&[],&[],&[],&[]],&[])?;
    if packed.iter().zip(&expected_packet).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err("wire BF16 packet differs from CPU twin".into());}
    // Wrong half order must fail on this input, including the signed-zero pattern.
    let mut wrong=expected.clone();wrong.swap(0,1);
    if got.iter().zip(wrong).all(|(a,b)|a.to_bits()==b.to_bits()) {return Err("wire wrong-order control did not fail".into());}
    let guard=b.read_vector_values(&mapped)?;
    if guard[..4].iter().chain(&guard[4+words/2..]).any(|v|v.to_bits()!=0x5a5aa5a5) {return Err("wire mapped guard changed".into());}
    // Destination aliases input: reject before any kernel can write it.
    if b.prompt_operation_output(pack,[Some(&source),None,None,None,None],None,Some(source.resident_range(0..words/2)?)).is_ok() {return Err("wire overlap control accepted".into());}
    Ok(words)
}
pub fn rejects(kind:usize)->Result<()> {
    if kind>=4 {return Err("invalid wire rejection control".into());}
    let (mut launched,mut refused)=(false,false);
    let result=spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<28,|b| {
        let mut values=vec![1.0;1024];values[if kind.is_multiple_of(2) {0} else {1023}]=f32::from_bits(if kind<2 {0x3f800001} else {0x7f800000});
        let source=b.upload_vector(&values)?;
        let pack=Shape {cols:1024,..Shape::new(Op::WirePackBf16,1).ok_or("rejection pack shape")?};
        let packet=b.prompt_operation(pack,[Some(&source),None,None,None,None],None)?;
        let _out=if kind>=2 {
            let s=Shape {rows:32,cols:32,first:u32::from(kind==3),slots:1,..Shape::new(Op::WireUnpackBf16,1).ok_or("rejection unpack shape")?};
            Some(b.prompt_operation(s,[Some(&packet),None,None,None,None],None)?)
        } else {None};
        launched=true;refused=b.synchronize().is_err();Ok(())
    });
    if !launched || !refused || result.is_ok() {return Err(format!("wire invalid-data control failed kind={kind} launched={launched} refused={refused} error={:?}",result.err()));}
    println!("WIRE REJECT PASS kind={kind} launched=true refused=true");Ok(())
}
pub fn run(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut cases=0;let mut words=0;let mut lines=String::new();
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,2<<30,|b| {
        let invalid=Shape {rows:2,cols:2,first:1,..Shape::new(Op::WireUnpackBf16,2).ok_or("bad control shape")?};
        if invalid.lengths().is_some() || b.prompt_operation(invalid,[None,None,None,None,None],None).is_ok() {return Err("wire transpose extra dimension accepted".into());}
        for queued in [false,true] {for (m,r,k) in [(1,2,8),(3,2,6),(8,4,32),(256,2,2048),(1024,2,2048),(2048,2,4096)] {for transpose in [false,true] {
            words+=run_case(b,m,r,k,transpose,queued)?;cases+=1;
        }}}
        for (m,k) in [(256,2048),(2048,2048),(2048,4096)] {
            let input=b.upload_vector(&vec![1.0;m*k])?;let destination=b.mapped_vector(m*k/2)?;
            let s=Shape {cols:(m*k) as u32,..Shape::new(Op::WirePackBf16,1).ok_or("timing shape")?};
            let mut samples=Vec::new();for _ in 0..8 {b.synchronize()?;let at=Instant::now();let packet=b.prompt_operation_output(s,[Some(&input),None,None,None,None],None,Some(destination.resident_range(0..destination.len())?))?;b.synchronize()?;samples.push(at.elapsed().as_secs_f64()*1e6);drop(packet);}
            samples.sort_by(f64::total_cmp);lines.push_str(&format!("WIRE PACK TIME words={} bytes={} median_us={:.3}\n",m*k,m*k*4,samples[4]));
        }Ok(())
    })?;
    println!("WIRE POSITIVE PASS cases={cases} words={words} packet_cpu_twin=true");
    // A device trap retires the CUDA process. Each wrong-input control gets its
    // own process, so a later control cannot pass or fail on an earlier fault.
    for kind in 0..4 {
        let child=std::process::Command::new(std::env::current_exe().map_err(|e|e.to_string())?).args(["wire-bf16-reject",&kind.to_string()]).output().map_err(|e|e.to_string())?;
        if !child.status.success() {return Err(format!("wire rejected-input child failed kind={kind}: {}",String::from_utf8_lossy(&child.stderr)));}
        let proof=String::from_utf8(child.stdout).map_err(|e|e.to_string())?;
        if !proof.contains(&format!("WIRE REJECT PASS kind={kind} launched=true refused=true")) {return Err("wire rejected-input child has no receipt".into());}
        lines.push_str(&proof);
    }
    lines.push_str(&format!("WIRE BF16 PASS cases={cases} words={words} differing_words=0 wrong_order_controls={cases} overlap_controls={cases} invalid_data_controls=4 guards=pass direct_and_recording_enabled_fallback=true actual_recorded=false packet_cpu_twin=true\n"));
    super::write(&out.join("COMPLETE"),lines.as_bytes())?;print!("{lines}");Ok(())
}
