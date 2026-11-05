//! Compare packed projection bytes across complete warps and partial row tails.
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{expert_ops::{self,Op,Shape},driver::with_backend};
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|e|e.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for cols in [16,32,128] {for rows in [1,15,16,17,31,32,33,65] {for case in 0..4 {
            let s=Shape {op:Op::Quantize,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75,activation:0.125,bound:0.0};
            let x=(0..cols as usize).map(|i|match case {1=>0.0,2=>-0.0,3=>match i%8 {0=>6.0,1=>0.0,2=>-0.0,3=>0.125,4=>-0.125,5=>f32::from_bits(1),6=>-f32::from_bits(1),_=>-0.25},_=>if i%7==0 {-0.0} else {(i as i32%19-9) as f32/8.0}}).collect::<Vec<_>>();
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
                super::write(&out.join(format!("r{rows}-k{cols}-case{case}-{repeat}.f32")),&super::floats(&got)?)?;
                receipt.push_str(&format!("NVFP4 DEVICE PASS rows={rows} cols={cols} case={case} repeat={repeat} quantized_bytes=true output_bits=true\n"));
            }
        }}}
        for cols in [16,128] {for rows in [1,9,33,65] {for count in [1,2,8,16] {
            let mut sources=Vec::new();let mut packed=Vec::new();
            for source in 0..3 {let shape=Shape {op:Op::Quantize,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75,activation:0.125*(source+1) as f32,bound:0.0};
                let input=(0..cols as usize).map(|index|if index%11==0 {-0.0} else {(index as i32%23-11) as f32*(source+1) as f32/8.0}).collect::<Vec<_>>();
                let input=b.upload_vector(&input)?;packed.push(b.expert_operation(shape,[Some(&input),None,None,None],None,None)?);sources.push(input);
            }
            let mut weights=Vec::new();let mut scales=Vec::new();let mut shapes=Vec::new();let mut expected=Vec::new();
            for job in 0..count {let shape=Shape {op:Op::NvProject,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.25*(job+1) as f32,activation:0.125*(job%3+1) as f32,bound:0.0};
                let bytes=(0..rows as usize*cols as usize/2).map(|index|((index*37+job*19+9)%256) as u8).collect::<Vec<_>>();
                weights.push(b.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())?);
                let bytes=(0..rows as usize*cols as usize/16).map(|index|0x20+((index+job)%48) as u8).collect::<Vec<_>>();
                scales.push(b.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())?);
                let scalar=b.expert_operation(shape,[Some(&packed[job%3]),None,None,None],Some(&weights[job]),Some(&scales[job]))?;
                expected.extend(b.read_vector_values(&scalar)?);shapes.push(shape);
            }
            let jobs=shapes.iter().enumerate().map(|(job,shape)|(*shape,&packed[job%3],&weights[job],&scales[job])).collect::<Vec<_>>();
            for repeat in 1..=3 {let output=b.expert_projection_batch(&jobs)?;let values=b.read_vector_values(&output)?;
                if bits(&values)!=bits(&expected) {return Err(format!("NVFP4 batch bits differ rows={rows} cols={cols} jobs={count} repeat={repeat}"));}
                super::write(&out.join(format!("batch-r{rows}-k{cols}-n{count}-{repeat}.f32")),&super::floats(&values)?)?;
                receipt.push_str(&format!("NVFP4 BATCH DEVICE PASS rows={rows} cols={cols} jobs={count} repeat={repeat} different_scales=true shared_inputs=true partial_rows=true output_bits=true\n"));
            }
            if b.expert_projection_batch(&[]).is_ok() {return Err("empty NVFP4 batch admitted".into());}
            let duplicate=b.expert_projection_batch(&[jobs[0];3])?;let duplicate=b.read_vector_values(&duplicate)?;
            if bits(&duplicate)!=bits(&expected[..rows as usize].repeat(3)) {return Err("repeated NVFP4 batch job differs".into());}
            let short=b.upload_vector(&vec![0.0;packed[0].len()-1])?;
            if b.expert_projection_batch(&[(shapes[0],&short,&weights[0],&scales[0])]).is_ok() {return Err("short NVFP4 batch source admitted".into());}
            let invalid=Shape {cols:cols+16,..shapes[0]};
            if b.expert_projection_batch(&[(invalid,&packed[0],&weights[0],&scales[0])]).is_ok() {return Err("wrong NVFP4 batch input admitted".into());}
            let invalid=Shape {rows:rows+1,..shapes[0]};
            if b.expert_projection_batch(&[jobs[0],(invalid,jobs[0].1,jobs[0].2,jobs[0].3)]).is_ok() {return Err("ragged NVFP4 batch admitted".into());}
        }}}
        Ok(())
    })?;super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
