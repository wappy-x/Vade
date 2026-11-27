//! Device packet-placement screen. The caller owns device access and saves its receipt.
#![forbid(unsafe_code)]
use super::{Result,write};
use std::path::Path;
use spark_gpu_probe::{driver::Backend,prompt_ops::Shape};
fn sample(rank:usize,token:usize,word:usize)->u32 {
    match word%19 {0=>0,1=>0x80000000,2=>1,3=>0x80000001,4=>0x7f7fffff,5=>0xff7fffff,
        _=>0x3f000000+((rank*104729+token*257+word)%0x7fffff) as u32}
}
fn check(b:&Backend<'_>,tokens:usize,width:usize,parts:usize,offset:usize)->Result<usize> {
    let mut input=vec![0.0;offset];
    for rank in 0..parts {for token in 0..tokens {for word in 0..width {input.push(f32::from_bits(sample(rank,token,word)));}}}
    let input=b.upload_vector(&input)?.into_resident_shared()?;
    let input=input.resident_range(offset..input.len())?;
    let shape=Shape::rank_rows(tokens,width,parts,true).ok_or("rank row screen shape")?;
    let output=b.prompt_operation(shape,[Some(&input),None,None,None,None],None)?;
    let got=b.read_vector_values(&output)?;
    for token in 0..tokens {for rank in 0..parts {for word in 0..width {
        let at=(token*parts+rank)*width+word;
        got[at].to_bits().eq(&sample(rank,token,word)).then_some(())
            .ok_or_else(||format!("rank row bits differ tokens={tokens} width={width} parts={parts} offset={offset} at={at}"))?;
    }}}
    Ok(got.len()*4)
}
pub fn run(out:&Path)->Result<()> {
    std::fs::create_dir(out).map_err(|e|e.to_string())?;let mut cases=0;let mut bytes=0;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for queued in [false,true] {
            let queue=if queued {Some(b.resident_queue()?)} else {None};
            for tokens in 1..=1024 {for width in [4,144] {bytes+=check(b,tokens,width,2,0)?;cases+=1;}}
            for parts in [1,2,3,4,6] {for tokens in [1,16,17,255,256,257,1024] {for width in [1,3,12,128,1024,2048,4096,8192] {
                bytes+=check(b,tokens,width,parts,0)?;cases+=1;
            }}}
            // An unaligned view takes the generic kernel, including whole vector widths.
            for offset in [1,2,3] {for tokens in [1,17,257] {for width in [3,4,144] {
                bytes+=check(b,tokens,width,2,offset)?;cases+=1;
            }}}
            drop(queue);
        }Ok(())
    })?;
    let line=format!("RANK ROWS PASS cases={cases} bytes={bytes} differing_bytes=0\n");write(&out.join("COMPLETE"),line.as_bytes())?;print!("{line}");Ok(())
}
