//! Compare packed projection bytes across complete warps and partial row tails.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{expert_ops::{self,Op,Shape},driver::with_backend};
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for cols in [16,32,128] {for rows in [1,15,16,17,31,32,33,65] {
            let s=Shape {op:Op::Quantize,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75,activation:0.125,bound:0.0};
            let x=(0..cols as usize).map(|i|if i%7==0 {-0.0} else {(i as i32%19-9) as f32/8.0}).collect::<Vec<_>>();
            let w=(0..rows as usize*cols as usize/2).map(|i|((i*37+9)%256) as u8).collect::<Vec<_>>();
            let scales=(0..rows as usize*cols as usize/16).map(|i|0x20+(i%48) as u8).collect::<Vec<_>>();
            let expected=expert_ops::reference(s,[&x,&[],&[],&[]],&[],&[])?;let source=b.upload_vector(&x)?;
            let weights=b.upload_vector(&w.chunks(4).map(|v| {let mut bytes=[0u8;4];bytes[..v.len()].copy_from_slice(v);f32::from_le_bytes(bytes)}).collect::<Vec<_>>())?;
            let scales_device=b.upload_vector(&scales.chunks(4).map(|v| {let mut bytes=[0u8;4];bytes[..v.len()].copy_from_slice(v);f32::from_le_bytes(bytes)}).collect::<Vec<_>>())?;
            for repeat in 1..=3 {
                let quantized=b.expert_operation(s,[Some(&source),None,None,None],None,None)?;let q=b.read_vector_values(&quantized)?;
                if bits(&q)!=bits(&expected) {return Err(format!("NVFP4 quantized bytes differ rows={rows} cols={cols}"));}
                let projection=Shape {op:Op::NvProject,..s};let expected=expert_ops::reference(projection,[&q,&[],&[],&[]],&w,&scales)?;
                let value=b.expert_operation(projection,[Some(&quantized),None,None,None],Some(&weights),Some(&scales_device))?;
                let got=b.read_vector_values(&value)?;
                if bits(&got)!=bits(&expected) {return Err(format!("NVFP4 projection bits differ rows={rows} cols={cols} repeat={repeat}"));}
                super::write(&out.join(format!("r{rows}-k{cols}-{repeat}.f32")),&super::floats(&got)?)?;
                receipt.push_str(&format!("NVFP4 DEVICE PASS rows={rows} cols={cols} repeat={repeat} quantized_bytes=true output_bits=true\n"));
            }
        }}Ok(())
    })?;super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
