//! Independent scalar-versus-grouped GPU row bits. No checkpoint or policy is selected.
#![forbid(unsafe_code)]
use super::{Result,write,floats,sha256};
use std::path::Path;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Shape,Op}};
fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {let mut v=b.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;b.write_vector_bytes(&mut v,bytes)?;b.synchronize()?;Ok(v)}
fn equal(a:&[f32],b:&[f32])->Result<()> {if a.len()!=b.len() || a.iter().zip(b).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err(format!("grouped prompt output differs from scalar bits lengths={}/{} first={:?}",a.len(),b.len(),a.iter().zip(b).enumerate().find(|(_, (a,b))|a.to_bits()!=b.to_bits()).map(|(at,(a,b))|(at,a.to_bits(),b.to_bits()))));}Ok(())}
fn shape(op:Op,rows:u32,cols:u32,tokens:u32)->Shape {Shape {op,rows,cols,tokens,block_rows:2,block_cols:16,global:0.75,activation:1.5,bound:0.0}}
fn packed(b:&Backend<'_>,s:Shape,output:&mut String)->Result<()> {
    let (r,k,m)=(s.rows as usize,s.cols as usize,s.tokens as usize);eprintln!("PROMPT FIXTURE op={:?} rows={r} cols={k} tokens={m}",s.op);
    let x=(0..m*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();
    let input=b.upload_vector(&x)?;let nv=s.op==Op::NvRows;
    let w=if nv {(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>()} else {(0..r*k).map(|i|(i%120+1) as u8).collect()};
    let scales=if nv {(0..r*k/16).map(|i|0x28+(i%24) as u8).collect::<Vec<_>>()} else {(0..r.div_ceil(2)*k.div_ceil(16)).flat_map(|i|(0.25+(i%9) as f32/16.0).to_le_bytes()).collect()};
    let weight=raw(b,&w)?;let scale=raw(b,&scales)?;
    let quantized=if nv {Some(b.expert_operation(Shape {op:Op::QuantizeRows,..s},[Some(&input),None,None,None],None,None)?)} else {None};
    let source=quantized.as_ref().unwrap_or(&input);let got=b.expert_operation(s,[Some(source),None,None,None],Some(&weight),Some(&scale))?;let got=b.read_vector_values(&got)?;
    let source_host=if nv {expert_ops::reference(Shape {op:Op::QuantizeRows,..s},[&x,&[],&[],&[]],&[],&[])?} else {x.clone()};
    let expected=expert_ops::reference(s,[&source_host,&[],&[],&[]],&w,&scales)?;equal(&got,&expected)?;
    for token in 0..m {let row=b.upload_vector(&x[token*k..(token+1)*k])?;
        let q=if nv {Some(b.expert_operation(Shape {op:Op::Quantize,tokens:1,..s},[Some(&row),None,None,None],None,None)?)} else {None};
        let scalar=b.expert_operation(Shape {op:if nv {Op::NvProject} else {Op::Fp8Project},tokens:1,..s},[Some(q.as_ref().unwrap_or(&row)),None,None,None],Some(&weight),Some(&scale))?;
        let scalar=b.read_vector_values(&scalar)?;let column=(0..r).map(|i|got[i*m+token]).collect::<Vec<_>>();equal(&scalar,&column)?;
    }
    output.push_str(&format!("op={:?} rows={r} cols={k} tokens={m} sha256={} scalar_bits=true host_bits=true\n",s.op,sha256::digest(&floats(&got)?)));Ok(())
}
fn plain(b:&Backend<'_>,m:usize,k:usize,output:&mut String)->Result<()> {
    use spark_gpu_probe::causal_ops::{self,Shape,Op};
    eprintln!("PROMPT FIXTURE op=ProjectRows rows=7 cols={k} tokens={m}");
    let s=Shape {op:Op::ProjectRows,heads:m as u32,dim:0,taps:0,rows:7,cols:k as u32,epsilon:0.0,bound:0.0};
    let x=(0..m*k).map(|i|if i%11==0 {-0.0} else {(i as i32%19-9) as f32/7.3}).collect::<Vec<_>>();let w=(0..7*k).map(|i|(i as i32%13-6) as f32/9.1).collect::<Vec<_>>();
    let input=b.upload_vector(&x)?;let weight=b.upload_vector(&w)?;let got=b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?;let got=b.read_vector_values(&got)?;
    equal(&got,&causal_ops::reference(s,[&x,&[],&[],&[]],&w)?).map_err(|e|format!("grouped versus host: {e}"))?;
    for token in 0..m {let row=b.upload_vector(&x[token*k..(token+1)*k])?;let scalar=b.causal_operation(Shape {op:Op::Project,..s},[Some(&row),None,None,None],Some(&weight))?;
        equal(&b.read_vector_values(&scalar)?,&(0..7).map(|r|got[r*m+token]).collect::<Vec<_>>()).map_err(|e|format!("scalar versus grouped token={token}: {e}"))?;}
    output.push_str(&format!("op=ProjectRows rows=7 cols={k} tokens={m} sha256={} scalar_bits=true host_bits=true\n",sha256::digest(&floats(&got)?)));Ok(())
}
/// Products whose fused multiply-add differs from a rounded product then add.
/// A contracted grouped kernel fails this against the one-row pinned kernel.
fn plain_unfused(b:&Backend<'_>,m:usize,output:&mut String)->Result<()> {
    use spark_gpu_probe::causal_ops::{self,Shape,Op};
    let a=f32::from_bits(0x3f800001);let c=f32::from_bits(0x3f7ffffe);
    let w=[-1.0,a,0.0,16777216.0,1.0,-16777216.0,-16777216.0,16777216.0,1.0];
    let x=(0..m).flat_map(|_|[1.0,c,1.0]).collect::<Vec<_>>();
    let s=Shape {op:Op::ProjectRows,heads:m as u32,dim:0,taps:0,rows:3,cols:3,epsilon:0.0,bound:0.0};
    let input=b.upload_vector(&x)?;let weight=b.upload_vector(&w)?;let got=b.read_vector_values(&b.causal_operation(s,[Some(&input),None,None,None],Some(&weight))?)?;
    equal(&got,&causal_ops::reference(s,[&x,&[],&[],&[]],&w)?).map_err(|e|format!("unfused grouped versus host: {e}"))?;
    if (0..m).any(|token|got[token].to_bits()!=0.0f32.to_bits()) {return Err("grouped plain projection fused a multiply and add".into());}
    for token in 0..m {let row=b.upload_vector(&x[token*3..token*3+3])?;let scalar=b.causal_operation(Shape {op:Op::Project,..s},[Some(&row),None,None,None],Some(&weight))?;
        equal(&b.read_vector_values(&scalar)?,&(0..3).map(|r|got[r*m+token]).collect::<Vec<_>>()).map_err(|e|format!("unfused scalar versus grouped token={token}: {e}"))?;}
    output.push_str(&format!("op=ProjectRows unfused rows=3 cols=3 tokens={m} sha256={} scalar_bits=true host_bits=true\n",sha256::digest(&floats(&got)?)));Ok(())
}
fn micro(b:&Backend<'_>,text:&mut String)->Result<()> {
    let s=shape(Op::NvRows,1024,4096,4);let (r,k,m)=(s.rows as usize,s.cols as usize,s.tokens as usize);
    let x=(0..m*k).map(|i|(i as i32%37-18) as f32/8.0).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
    let q=b.expert_operation(Shape {op:Op::QuantizeRows,..s},[Some(&input),None,None,None],None,None)?;
    let weight=raw(b,&(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>())?;let scale=raw(b,&vec![0x38;r*k/16])?;
    let mut scalar_inputs=Vec::new();for token in 0..m {scalar_inputs.push(b.upload_vector(&expert_ops::reference(Shape {op:Op::Quantize,tokens:1,..s},[&x[token*k..(token+1)*k],&[],&[],&[]],&[],&[])?)?);}
    b.set_profiling(true);
    for repeat in 1..=3 {let before=b.stats();let mut expected=Vec::new();for row in &scalar_inputs {expected.push(b.expert_operation(Shape {op:Op::NvProject,tokens:1,..s},[Some(row),None,None,None],Some(&weight),Some(&scale))?);}
        let scalar=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();let before=b.stats();
        let grouped=b.expert_operation(s,[Some(&q),None,None,None],Some(&weight),Some(&scale))?;let grouped_ns=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();
        let got=b.read_vector_values(&grouped)?;for (token,row) in expected.iter().enumerate() {equal(&b.read_vector_values(row)?,&(0..r).map(|row|got[row*m+token]).collect::<Vec<_>>())?;}
        text.push_str(&format!("MICRO repeat={repeat} rows={r} cols={k} tokens={m} scalar_gpu_ns={scalar} grouped_gpu_ns={grouped_ns} scalar_seconds_per_row={:.9} grouped_seconds_per_row={:.9} exact=true\n",scalar as f64/1e9/m as f64,grouped_ns as f64/1e9/m as f64));
    }b.set_profiling(false);Ok(())
}
fn micro_fp8(b:&Backend<'_>,text:&mut String)->Result<()> {
    let s=Shape {block_rows:128,block_cols:128,..shape(Op::Fp8Rows,1024,4096,4)};
    let (r,k,m)=(s.rows as usize,s.cols as usize,s.tokens as usize);
    let x=(0..m*k).map(|i|(i as i32%37-18) as f32/8.0).collect::<Vec<_>>();let input=b.upload_vector(&x)?;
    let weight=raw(b,&(0..r*k).map(|i|(i%120+1) as u8).collect::<Vec<_>>())?;
    let scale=raw(b,&vec![1.0f32;r.div_ceil(128)*k.div_ceil(128)].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
    let inputs=(0..m).map(|token|b.upload_vector(&x[token*k..(token+1)*k])).collect::<Result<Vec<_>>>()?;
    b.set_profiling(true);
    for repeat in 1..=3 {let before=b.stats();let mut expected=Vec::new();for row in &inputs {
        expected.push(b.expert_operation(Shape {op:Op::Fp8Project,tokens:1,..s},[Some(row),None,None,None],Some(&weight),Some(&scale))?);}
        let scalar=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();let before=b.stats();
        let grouped=b.expert_operation(s,[Some(&input),None,None,None],Some(&weight),Some(&scale))?;
        let grouped_ns=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();let got=b.read_vector_values(&grouped)?;
        for (token,row) in expected.iter().enumerate() {equal(&b.read_vector_values(row)?,&(0..r).map(|row|got[row*m+token]).collect::<Vec<_>>())?;}
        text.push_str(&format!("MICRO_FP8 repeat={repeat} rows={r} cols={k} tokens={m} scalar_gpu_ns={scalar} grouped_gpu_ns={grouped_ns} exact=true\n"));
    }b.set_profiling(false);Ok(())
}
pub(super) fn run(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut bits=String::new();let mut timing=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,512<<20,|b| {
        for repeat in 1..=3 {bits.push_str(&format!("repeat={repeat}\n"));for m in 1..=4 {for k in [16,32,96] {packed(b,shape(Op::NvRows,33,k,m),&mut bits)?;}
            packed(b,shape(Op::Fp8Rows,5,33,m),&mut bits)?;for k in [1,17,128] {plain(b,m as usize,k,&mut bits)?;}plain_unfused(b,m as usize,&mut bits)?;}}
        micro(b,&mut timing)?;micro_fp8(b,&mut timing)
    })?;
    write(&out.join("BITS"),bits.as_bytes())?;write(&out.join("TIMING"),timing.as_bytes())?;write(&out.join("COMPLETE"),b"PROMPT ROW DEVICE PASS\n")?;print!("{timing}");println!("PROMPT ROW DEVICE PASS");Ok(())
}
