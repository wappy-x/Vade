//! Native NV row geometry screen. Saved bytes compare independent builds.
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Op,Shape}};
fn raw(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect()}
fn upload<'a>(backend:&Backend<'a>,bytes:&[u8])->Result<DeviceVector<'a>> {
    backend.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())
}
fn mixed(index:usize,job:usize)->u32 {
    let mut value=(index as u32).wrapping_add((job as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=format!("NV_GROUP native_rows={} physical_threads=32 timing=complete_public_call_including_allocation_zero_and_table conditioning_outside_timer=true\n",expert_ops::NV_ROWS_PER_WARP);
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let cache=backend.upload_vector(&vec![1.0;64<<20])?;
        for (rows,cols,max_jobs) in [(1u32,16u32,16usize),(7,32,16),(8,48,16),(9,64,16),(15,128,16),(16,16,16),(17,32,16),(31,48,16),(32,64,16),(33,128,16),(65,16,16),(1024,4096,16),(2048,2048,8)] {
            let (mut shapes,mut inputs,mut weights,mut scales,mut expected)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
            for job in 0..max_jobs {
                let shape=Shape {op:Op::NvProject,rows,cols,block_rows:1,block_cols:16,tokens:1,global:0.75+job as f32/16.0,activation:0.125+job as f32/32.0,bound:0.0};
                let source=(0..cols as usize).map(|index|match (index+job)%13 {0=>0.0,1=>-0.0,2=>f32::from_bits(1),3=>-f32::from_bits(1),_=>((mixed(index,job)%47) as i32-23) as f32/17.0}).collect::<Vec<_>>();
                let quantized=expert_ops::reference(Shape {op:Op::Quantize,..shape},[&source,&[],&[],&[]],&[],&[])?;
                let bytes=(0..rows as usize*cols as usize/2).map(|index|mixed(index,job) as u8).collect::<Vec<_>>();
                let scale=(0..rows as usize*cols as usize/16).map(|index|if index%17==0 {0} else {0x28+(mixed(index,job+31)%64) as u8}).collect::<Vec<_>>();
                expected.push(expert_ops::reference(shape,[&quantized,&[],&[],&[]],&bytes,&scale)?);
                inputs.push(backend.upload_vector(&quantized)?);weights.push(upload(backend,&bytes)?);scales.push(upload(backend,&scale)?);shapes.push(shape);
            }
            let jobs=(0..max_jobs).map(|job|(shapes[job],&inputs[job],&weights[job],&scales[job])).collect::<Vec<_>>();
            for count in [1,2,8,16].into_iter().filter(|&count|count<=max_jobs) {
                let expected=raw(&expected[..count].concat());
                for repeat in 1..=3 {
                    let output=backend.expert_projection_batch(&jobs[..count])?;let actual=raw(&backend.read_vector_values(&output)?);
                    if actual!=expected {return Err(format!("NV GROUP BITS DIFFER rows={rows} cols={cols} jobs={count} repeat={repeat}"));}
                    super::write(&out.join(format!("batch-r{rows}-k{cols}-j{count}-{repeat}.f32")),&actual)?;
                }
                if rows>=1024 {for cold in [false,true] {for sample in 1..=5 {
                    if cold {let conditioned=backend.round_bf16_lanes(&cache)?;backend.synchronize()?;drop(conditioned);}
                    backend.synchronize()?;let started=Instant::now();let output=backend.expert_projection_batch(&jobs[..count])?;
                    let ns=started.elapsed().as_nanos();if raw(&backend.read_vector_values(&output)?)!=expected {return Err("timed NV group bytes differ".into());}
                    receipt.push_str(&format!("NV_GROUP_MICRO rows={rows} cols={cols} jobs={count} cold_cache={cold} sample={sample} ns={ns} conditioning_bytes=268435456 exact=true\n"));
                }}}
            }
            for job in 0..2 {let (shape,input,weight,scales)=jobs[job];
                let output=backend.expert_operation(shape,[Some(input),None,None,None],Some(weight),Some(scales))?;
                let actual=raw(&backend.read_vector_values(&output)?);if actual!=raw(&expected[job]) {return Err("single NV group bytes differ".into());}
                super::write(&out.join(format!("single-r{rows}-k{cols}-j{job}.f32")),&actual)?;
            }
            receipt.push_str(&format!("NV_GROUP_PASS rows={rows} cols={cols} max_jobs={max_jobs} repeats=3 raw_bits=true distinct_rows_and_jobs=true\n"));
            super::write(&out.join(format!("RECEIPT-r{rows}-k{cols}")),receipt.as_bytes())?;
        }
        edges(backend,out,&mut receipt)?;all_codes(backend,out,&mut receipt)
    })?;super::write(&out.join("RECEIPT"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
fn edges(backend:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    for global in [1,0x0000_8000,0x0001_8000,0x007f_ffff,0x0080_0000,0x3f80_8000,0x3f81_8000,0x7f7f_0000,0x7f7f_ffff] {for case in 0..4 {
        let shape=Shape {op:Op::NvProject,rows:33,cols:16,block_rows:1,block_cols:16,tokens:1,global:f32::from_bits(global),activation:1.0,bound:0.0};
        let mut source=vec![if case==1 {-0.0} else {0.0};17];source[16]=1.0;
        if case>=2 {source[0]=if case==3 {6.0} else {1.0};source[1]=if case==3 {6.0} else {-0.0};}
        let mut weight=vec![0;33*8];for row in 0..33 {weight[row*8]=if case==3 {0xa2} else {0x02};}let scales=vec![0x38;33];
        let host=match expert_ops::reference(shape,[&source,&[],&[],&[]],&weight,&scales) {
            Ok(values)=>Some(raw(&values)),Err(error) if error=="expert operation output is not finite"=>None,Err(error)=>return Err(error),
        };
        let input=backend.upload_vector(&source)?;let weight=upload(backend,&weight)?;let scales=upload(backend,&scales)?;let jobs=[(shape,&input,&weight,&scales);2];let mut previous=None;
        for repeat in 1..=3 {
            let output=backend.expert_projection_batch(&jobs)?;let bytes=raw(&backend.read_vector_values(&output)?);
            if let Some(host)=&host&& bytes!=host.repeat(2) {return Err(format!("NV group finite edge differs global={global:08x} case={case}"));}
            if let Some(previous)=&previous {if previous!=&bytes {return Err("NV group edge repeat differs".into());}}else {previous=Some(bytes.clone());}
            super::write(&out.join(format!("edge-{global:08x}-case{case}-{repeat}.f32")),&bytes)?;
        }
        receipt.push_str(&format!("NV_GROUP_EDGE global_bits={global:08x} case={case} host_finite={} repeats=3 exact=true\n",host.is_some()));
    }}Ok(())
}
pub(super) fn all_codes(backend:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let shape=Shape {op:Op::NvProject,rows:65536,cols:16,block_rows:1,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0};
    let bytes=(0..65536usize).flat_map(|row|[row as u8;8]).collect::<Vec<_>>();
    let scale=(0..65536usize).map(|row|(row/256) as u8).collect::<Vec<_>>();
    let weight=upload(backend,&bytes)?;let scales=upload(backend,&scale)?;
    for pattern in [0,0x8000_0000,1,0x8000_0001,0x0080_0000,0x3f80_8000,0x3f81_8000,0x7f7f_0000,0x7f80_0000,0xff80_0000,0x7fc0_1234] {
        let mut source=vec![f32::from_bits(pattern);17];source[16]=1.0;let input=backend.upload_vector(&source)?;let mut previous=None;
        for repeat in 1..=3 {
            let output=backend.expert_operation(shape,[Some(&input),None,None,None],Some(&weight),Some(&scales))?;
            let actual=backend.read_vector_values(&output)?;
            for (row,&value) in actual.iter().enumerate() {let mut expected=[0.0];expert_ops::unit(shape,row,[&source,&[],&[],&[]],&bytes,&scale,&mut expected);
                if expected[0].is_finite() && value.to_bits()!=expected[0].to_bits() {return Err(format!("NV all-code finite row differs input={pattern:08x} row={row}"));}
            }
            let bytes=raw(&actual);if let Some(previous)=&previous {if previous!=&bytes {return Err("NV all-code repeated raw bits differ".into());}} else {previous=Some(bytes.clone());}
            super::write(&out.join(format!("allcodes-input{pattern:08x}-{repeat}.f32")),&bytes)?;
        }
        receipt.push_str(&format!("NV_GROUP_ALL_CODES input_bits={pattern:08x} paired_weight_codes=256 scale_codes=256 repeats=3 finite_host_bits=true nonfinite_raw_bytes_saved=true\n"));
    }Ok(())
}
