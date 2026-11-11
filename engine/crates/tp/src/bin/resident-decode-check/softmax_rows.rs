//! Softmax shape sweep. A 65-head generic launch is an independent
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
        let value=match pattern {1=>if at.is_multiple_of(2) {-0.0} else {0.0},2=>if slot==k/2 {80.0} else {-80.0},3=>f32::from_bits((at%7) as u32),_=>((at*17%257) as i32-128) as f32/13.0};
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
        for m in 1..=16 {for k in 1..=SLOTS as usize {case(b,m,k,0)?;comparisons+=1;if k%64==0 {b.release_group_spares()?;}}
            println!("SOFTMAX ROW SWEEP tokens={m} widths=1..={SLOTS} comparisons={comparisons}");}
        for m in [1,2,3,4,16] {for k in [1,2,31,32,33,127,128,129,1040,1152,2048,2051] {for pattern in 1..=3 {case(b,m,k,pattern)?;comparisons+=1;}}}
        Ok(())
    })?;
    let calls=[count("prompt_softmax_rows")?-before[0],count("prompt_rows")?-before[1]];
    if calls!=[comparisons,comparisons] {return Err(format!("softmax paths did not run: {calls:?}, wanted {comparisons}"));}
    let line=format!("SOFTMAX ROW BYTES PASS comparisons={comparisons} dense_shapes={} edge_cases=180 tokens_max=16 heads=64 fallback_heads=65 poisoned_output=true kernel_calls={} reference_calls={}\n",16*SLOTS,calls[0],calls[1]);
    write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}

/// Prompt groups: every new token count at both width limits and a partial
/// warp. Additional widths and nonzero patterns test the group boundaries.
pub(super) fn prompt(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut comparisons=0;
    let before=[count("prompt_softmax_rows")?,count("prompt_rows")?];
    with_backend(0,spark_gpu_probe::layer_ptx()?,2<<30,|b|{
        b.configure_vector_pool(1<<30);
        for m in 17..=MAX_TOKENS as usize {for k in [1,33,SLOTS as usize] {case(b,m,k,0)?;comparisons+=1;}b.release_group_spares()?;}
        for m in [17,31,32,62,63,64,127,128,255,256] {for k in [2,31,32,127,128,129,1040,1152,2048] {for pattern in 1..=3 {case(b,m,k,pattern)?;comparisons+=1;}}b.release_group_spares()?;}
        Ok(())
    })?;
    let calls=[count("prompt_softmax_rows")?-before[0],count("prompt_rows")?-before[1]];
    if calls!=[comparisons,comparisons] {return Err(format!("prompt softmax paths did not run: {calls:?}, wanted {comparisons}"));}
    let line=format!("PROMPT SOFTMAX BYTES PASS comparisons={comparisons} tokens=17..={MAX_TOKENS} width_axes=1,33,{SLOTS} edge_cases=270 heads=64 fallback_heads=65 poisoned_output=true kernel_calls={} reference_calls={} full_cartesian_sweep=false\n",calls[0],calls[1]);
    print!("{line}");let mut timing=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,2<<30,|b|{
        b.configure_vector_pool(1<<30);b.set_profiling(true);
        for (m,k) in [(256usize,256usize),(256,1040),(256,2051),(62,2051)] {
            let s=Shape {heads:HEADS,cols:k as u32,slots:SLOTS,..Shape::new(Op::SoftmaxRows,m).ok_or("prompt timing shape")?};
            let mut selected=vec![0.0;m*(SLOTS as usize+1)];for token in 0..m {selected[token*(SLOTS as usize+1)+SLOTS as usize]=k as f32;}
            let ids=b.upload_vector(&selected)?;
            let x=b.upload_vector(&(0..m*HEADS as usize*k).map(|at|((at*17%257) as i32-128) as f32/13.0).collect::<Vec<_>>())?;
            let old=b.upload_vector(&(0..m*(HEADS as usize+1)*k).map(|at|((at*17%257) as i32-128) as f32/13.0).collect::<Vec<_>>())?;
            for sample in 0..7 {for generic in if sample%2==0 {[true,false]} else {[false,true]} {
                let shape=if generic {Shape {heads:HEADS+1,..s}} else {s};let input=if generic {&old} else {&x};
                let before=b.stats();let started=std::time::Instant::now();
                let value=b.prompt_operation(shape,[Some(input),Some(&ids),None,None,None],None)?;b.synchronize()?;
                let ns=b.stats().since(before).device_kernel_ns[..14].iter().sum::<u128>();
                timing.push_str(&format!("PROMPT SOFTMAX TIME tokens={m} cols={k} generic={generic} heads={} sample={sample} gpu_ns={ns} wall_ns={}\n",shape.heads,started.elapsed().as_nanos()));drop(value);
            }}
        }Ok(())
    })?;
    write(&out.join("TIMING"),timing.as_bytes())?;print!("{timing}");write(&out.join("COMPLETE"),line.as_bytes())?;Ok(())
}
