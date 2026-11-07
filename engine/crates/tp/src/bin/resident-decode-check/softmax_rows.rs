//! Full admitted softmax shape sweep. A 65-head generic launch is an independent
//! oracle: the extra zero head cannot affect any of the first 64 head rows.
#![forbid(unsafe_code)]
use super::{Result,write};
use std::path::Path;
use spark_gpu_probe::{driver::with_backend,prompt_ops::{Op,Shape},softmax_rows::{MAX_TOKENS,HEADS,SLOTS}};
fn case(b:&spark_gpu_probe::driver::Backend<'_>,m:usize,k:usize,pattern:usize)->Result<()> {
    let h=HEADS as usize;let slots=SLOTS as usize;
    let mut input=Vec::with_capacity(m*h*k);let mut oracle=Vec::with_capacity(m*(h+1)*k);
    for token in 0..m {for head in 0..h {for slot in 0..k {
        let at=(token*h+head)*k+slot;
        let value=match pattern {1=>if at%2==0 {-0.0} else {0.0},2=>if slot==k/2 {80.0} else {-80.0},3=>f32::from_bits((at%7) as u32),_=>((at*17%257) as i32-128) as f32/13.0};
        input.push(value);oracle.push(value);
    }}oracle.resize(oracle.len()+k,0.0);}
    let mut selected=vec![0.0;m*(slots+1)];
    for token in 0..m {let choices=[0,1,k.saturating_sub(1),k];selected[token*(slots+1)+slots]=if m==1 {k} else {choices[(token+pattern)%4]} as f32;}
    let (x,old,ids)=(b.upload_vector(&input)?,b.upload_vector(&oracle)?,b.upload_vector(&selected)?);
    let s=Shape {heads:HEADS,cols:k as u32,slots:SLOTS,..Shape::new(Op::SoftmaxRows,m).ok_or("softmax sweep shape")?};
    // Keep only this poisoned spare. Address equality proves that the no-fill
    // kernel writes every result word over 0xff, including zero-count tails.
    b.release_group_spares()?;
    let poison=b.upload_vector(&vec![f32::from_bits(0xffff_ffff);m*h*k])?;
    let poisoned=poison.device_address();b.synchronize()?;drop(poison);
    let got=b.prompt_operation(s,[Some(&x),Some(&ids),None,None,None],None)?;
    if got.device_address()!=poisoned {return Err("softmax output did not reuse the poisoned allocation".into());}
    // 65 is outside admission and therefore MUST run the original prompt_rows.
    let expected=b.prompt_operation(Shape {heads:HEADS+1,..s},[Some(&old),Some(&ids),None,None,None],None)?;
    let (got,expected)=(b.read_vector_values(&got)?,b.read_vector_values(&expected)?);
    for token in 0..m {for at in 0..h*k {
        if got[token*h*k+at].to_bits()!=expected[token*(h+1)*k+at].to_bits() {
            return Err(format!("softmax bytes differ tokens={m} cols={k} pattern={pattern} token={token} at={at}"));
        }
    }}Ok(())
}
fn count(name:&str)->Result<u64> {spark_gpu_probe::prompt_launches::line().split_whitespace().find_map(|word|word.strip_prefix(&format!("{name}="))).ok_or("softmax counter absent")?.parse().map_err(|e:std::num::ParseIntError|e.to_string())}
pub(super) fn run(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut comparisons=0;
    let before=[count("prompt_softmax_rows")?,count("prompt_rows")?];
    with_backend(0,spark_gpu_probe::layer_ptx()?,2<<30,|b|{
        b.configure_vector_pool(1<<30);
        for m in 1..=MAX_TOKENS as usize {for k in 1..=SLOTS as usize {case(b,m,k,0)?;comparisons+=1;if k%64==0 {b.release_group_spares()?;}}
            println!("SOFTMAX ROW SWEEP tokens={m} widths=1..={SLOTS} comparisons={comparisons}");}
        for m in [1,2,3,4,16] {for k in [1,2,31,32,33,127,128,129,1040,1152,2048,2051] {for pattern in 1..=3 {case(b,m,k,pattern)?;comparisons+=1;}}}
        Ok(())
    })?;
    let calls=[count("prompt_softmax_rows")?-before[0],count("prompt_rows")?-before[1]];
    if calls!=[comparisons,comparisons] {return Err(format!("softmax paths did not run: {calls:?}, wanted {comparisons}"));}
    let line=format!("SOFTMAX ROW BYTES PASS comparisons={comparisons} dense_shapes={} edge_cases=180 heads=64 fallback_heads=65 poisoned_output=true kernel_calls={} reference_calls={}\n",MAX_TOKENS*SLOTS,calls[0],calls[1]);
    write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
