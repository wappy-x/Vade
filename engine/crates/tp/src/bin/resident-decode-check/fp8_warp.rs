use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Op,Shape},causal_ops};
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|value|value.to_bits()).collect()}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let cache=backend.upload_vector(&vec![1.0;64<<20])?;
        for (rows,cols) in [(1u32,1u32),(7,2),(9,3),(7,4),(9,5),(9,6),(33,7),(9,16),(7,17),(33,32),(9,48),(9,64),(33,96),(9,33),(33,129),(1024,4096),(4096,1024),(2048,16384)] {
            let widths=if rows<100 {vec![1,2,3,5,7,16,128]} else {vec![128]};
            for block_width in widths {
            let source=(0..cols).map(|index|match index%13 {0=>-0.0,1=>16_777_216.0,2=>-16_777_216.0,3=>f32::from_bits(0x00018000),_=>(index as i32%23-11) as f32/17.0}).collect::<Vec<_>>();
            let input=backend.upload_vector(&source)?;let mut weights=Vec::new();let mut scales=Vec::new();let mut shapes=Vec::new();let mut expected=Vec::new();
            for job in 0..2 {let shape=Shape {op:Op::Fp8Project,rows,cols,block_rows:if rows<100 && job==1 {3} else {128},block_cols:if rows<100 && job==1 {block_width} else {128},tokens:1,global:1.0,activation:1.0,bound:0.0};
                let bytes=(0..rows as usize*cols as usize).map(|index|{let magnitude=if rows<100 {((index*53+job*17)%127) as u8} else {0x20+((index*7+job*3)%79) as u8};magnitude|if index%5==0 {128} else {0}}).collect::<Vec<_>>();
                let scale=(0..rows.div_ceil(shape.block_rows)*cols.div_ceil(shape.block_cols)).flat_map(|index|{
                    let value=if rows<100 && index%5==0 {f32::from_bits(0x00018000)} else {0.03125*(1+(index+job as u32)%31) as f32};value.to_le_bytes()
                }).collect::<Vec<_>>();
                let host=expert_ops::reference(shape,[&source,&[],&[],&[]],&bytes,&scale)?;
                let upload=|bytes:&[u8]|backend.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>());
                weights.push(upload(&bytes)?);scales.push(upload(&scale)?);shapes.push(shape);expected.extend(host);
            }
            let scalar=shapes.iter().enumerate().map(|(job,shape)|(*shape,&input,&weights[job],&scales[job])).collect::<Vec<_>>();
            let serial=scalar.iter().map(|&(shape,input,weights,scales)|(Shape {op:Op::Fp8Serial,..shape},input,weights,scales)).collect::<Vec<_>>();
            for repeat in 1..=3 {for (job,&(shape,input,weights,scales)) in serial.iter().enumerate() {
                let output=backend.expert_operation(shape,[Some(input),None,None,None],Some(weights),Some(scales))?;
                let values=backend.read_vector_values(&output)?;
                if bits(&values)!=bits(&expected[job*rows as usize..(job+1)*rows as usize]) {return Err(format!("single serial FP8 bits differ rows={rows} cols={cols} job={job} repeat={repeat}"));}
                super::write(&out.join(format!("single-serial-r{rows}-k{cols}-bc{block_width}-j{job}-{repeat}.f32")),&super::floats(&values)?)?;
            }}
            for repeat in 1..=3 {for (name,jobs) in [("mapped",&scalar),("serial",&serial)] {
                let output=backend.expert_projection_batch(jobs)?;let values=backend.read_vector_values(&output)?;
                if bits(&values)!=bits(&expected) {let row=values.iter().zip(&expected).position(|(left,right)|left.to_bits()!=right.to_bits()).ok_or("missing differing row")?;
                    return Err(format!("FP8 SERIAL BITS DIFFER schedule={name} rows={rows} cols={cols} repeat={repeat} row={row} device={:08x} host={:08x}",values[row].to_bits(),expected[row].to_bits()));
                }
                super::write(&out.join(format!("{name}-r{rows}-k{cols}-bc{block_width}-{repeat}.f32")),&super::floats(&values)?)?;
            }}
            let duplicate=backend.expert_projection_batch(&[serial[0];3])?;
            if bits(&backend.read_vector_values(&duplicate)?)!=bits(&expected[..rows as usize].repeat(3)) {return Err("repeated serial FP8 job differs".into());}
            if rows>=1024 {for count in [1,2,8] {
                let mapped=(0..count).map(|index|scalar[index%2]).collect::<Vec<_>>();let serial=(0..count).map(|index|serial[index%2]).collect::<Vec<_>>();
                let expected=(0..count).flat_map(|index|expected[index%2*rows as usize..(index%2+1)*rows as usize].iter().copied()).collect::<Vec<_>>();
                for cold in [false,true] {for sample in 1..=5 {
                let mut timings=[0u128;2];
                for index in if sample%2==0 {[0usize,1]} else {[1,0]} {
                    if cold {let conditioned=backend.round_bf16_lanes(&cache)?;backend.synchronize()?;drop(conditioned);}
                    backend.synchronize()?;let start=Instant::now();let output=backend.expert_projection_batch(if index==0 {&mapped} else {&serial})?;
                    timings[index]=start.elapsed().as_nanos();
                    if bits(&backend.read_vector_values(&output)?)!=bits(&expected) {return Err("timed serial FP8 bytes differ".into());}
                }
                receipt.push_str(&format!("FP8_SERIAL_MICRO rows={rows} cols={cols} jobs={count} block_rows=128 block_cols=128 sample={sample} mapped_batch_ns={} serial_batch_ns={} cold_cache={cold} conditioning_bytes=268435456 alternating_order=true exact=true\n",timings[0],timings[1]));
            }}}}
            receipt.push_str(&format!("FP8 SERIAL DEVICE PASS rows={rows} cols={cols} repeats=3 jobs=2 repeated_jobs=3 partial_rows=true partial_features=true ordered_adds=true exact_bits=true\n"));
            }
        }
        plain(backend,&cache,out,&mut receipt)?;
        edge_bits(backend,out,&mut receipt)?;
        Ok(())
    })?;
    super::write(&out.join("RECEIPT"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
fn edge_bits(backend:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let patterns=[0,0x8000_0000,1,0x8000_0001,0x0000_7fff,0x0000_8000,0x0001_8000,0x007f_ffff,0x0080_0000,0x8080_0000,0x3f80_7fff,0x3f80_8000,0x3f81_8000,0x7f7f_0000,0xff7f_0000,0x7f7f_ffff];
    for cols in [16u32,17,32,33,128] {for case in 0..4 {
        let source=(0..cols as usize).map(|index|f32::from_bits(match case {1=>patterns[index%2],2=>patterns[13+index%2],_=>patterns[index%patterns.len()]})).collect::<Vec<_>>();
        let shape=Shape {op:Op::Fp8Project,rows:33,cols,block_rows:3,block_cols:if case==3 {3} else {128},tokens:1,global:1.0,activation:1.0,bound:0.0};
        let bytes=(0..shape.rows as usize*cols as usize).map(|index|if case==1 {if index%2==0 {0} else {128}} else {((index*53)%127) as u8|if index%5==0 {128} else {0}}).collect::<Vec<_>>();
        let scale=vec![if case==3 {f32::MIN_POSITIVE} else {1.0};shape.rows.div_ceil(shape.block_rows) as usize*cols.div_ceil(shape.block_cols) as usize];
        let input=backend.upload_vector(&source)?;
        let weight=backend.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())?;
        let scales=backend.upload_vector(&scale)?;
        for repeat in 1..=3 {
            let before=backend.expert_operation(shape,[Some(&input),None,None,None],Some(&weight),Some(&scales))?;
            let after=backend.expert_operation(Shape {op:Op::Fp8Serial,..shape},[Some(&input),None,None,None],Some(&weight),Some(&scales))?;
            let before=backend.read_vector_values(&before)?;let after=backend.read_vector_values(&after)?;
            if bits(&before)!=bits(&after) {let row=bits(&before).iter().zip(bits(&after)).position(|(left,right)|*left!=right).ok_or("missing edge difference")?;return Err(format!("FP8 SERIAL EDGE BITS DIFFER cols={cols} case={case} row={row} old={:08x} new={:08x}",before[row].to_bits(),after[row].to_bits()));}
            super::write(&out.join(format!("edges-k{cols}-case{case}-{repeat}.f32")),&after.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect::<Vec<_>>())?;
        }
        receipt.push_str(&format!("FP8 SERIAL EDGE PASS rows=33 cols={cols} case={case} repeats=3 raw_bits=true\n"));
    }}Ok(())
}
fn plain(backend:&Backend<'_>,cache:&DeviceVector<'_>,out:&Path,receipt:&mut String)->Result<()> {
    for (rows,cols) in [(1u32,1u32),(7,3),(33,4),(9,7),(33,16),(7,17),(9,33),(33,32),(33,64),(256,4096),(1024,4096),(16384,512)] {
        let source=(0..cols).map(|index|match index%11 {0=>-0.0,1=>f32::from_bits(1),2=>f32::MIN_POSITIVE,_=>(index as i32%23-11) as f32/17.0}).collect::<Vec<_>>();
        let input=backend.upload_vector(&source)?;let values=(0..rows as usize*cols as usize).map(|index|match index%13 {0=>-0.0,1=>0.0,2=>f32::from_bits(0x18000),_=>((index*29)%101) as f32/128.0-0.25}).collect::<Vec<_>>();
        let weight=backend.upload_vector(&values)?;
        let baseline=causal_ops::Shape {op:causal_ops::Op::Project,rows,cols,heads:0,dim:0,taps:0,epsilon:0.0,bound:0.0};
        let shape=Shape {op:Op::PlainSerial,rows,cols,block_rows:0,block_cols:0,tokens:1,global:1.0,activation:1.0,bound:0.0};
        let expected=causal_ops::reference(baseline,[&source,&[],&[],&[]],&values)?;
        let production=backend.causal_operation(baseline,[Some(&input),None,None,None],Some(&weight))?;
        if bits(&backend.read_vector_values(&production)?)!=bits(&expected) {return Err("production plain bits differ from scalar reference".into());}
        for jobs in [1,2,8] {for repeat in 1..=3 {
            let output=backend.expert_projection_batch(&vec![(shape,&input,&weight,&weight);jobs])?;let actual=backend.read_vector_values(&output)?;
            if bits(&actual)!=bits(&expected.repeat(jobs)) {return Err(format!("PLAIN SERIAL BITS DIFFER rows={rows} cols={cols} jobs={jobs}"));}
            super::write(&out.join(format!("plain-r{rows}-k{cols}-j{jobs}-{repeat}.f32")),&super::floats(&actual)?)?;
        }}
        if cols%32==0 {for count in [1,2,8] {for round in [false,true] {
            let jobs=(0..count).map(|_|spark_gpu_probe::driver::ResidentProjection::Plain {rows:rows as usize,cols:cols as usize,weight:&weight,input:&input}).collect::<Vec<_>>();
            let mut output=backend.vector(rows as usize*count)?;backend.resident_projection_batch(&jobs,&mut output,round)?;
            let expected=expected.iter().map(|&value|if round {expert_ops::round(value)} else {value}).collect::<Vec<_>>().repeat(count);
            let actual=backend.read_vector_values(&output)?;if bits(&actual)!=bits(&expected) {return Err("production plain batch bits differ from scalar reference".into());}
            super::write(&out.join(format!("production-plain-r{rows}-k{cols}-j{count}-round{round}.f32")),&super::floats(&actual)?)?;
        }}}
        if rows>=256 {
            for cold in [false,true] {for sample in 1..=5 {
                let mut times=[0u128;2];for variant in if sample%2==0 {[0usize,1]} else {[1,0]} {
                    if cold {let conditioned=backend.round_bf16_lanes(cache)?;backend.synchronize()?;drop(conditioned);}
                    backend.synchronize()?;let started=Instant::now();let output=if variant==0 {backend.causal_operation(baseline,[Some(&input),None,None,None],Some(&weight))?}
                        else {backend.expert_operation(shape,[Some(&input),None,None,None],Some(&weight),None)?};
                    times[variant]=started.elapsed().as_nanos();let actual=backend.read_vector_values(&output)?;
                    if bits(&actual)!=bits(&expected) {return Err("timed plain serial bits differ".into());}
                }
                receipt.push_str(&format!("PLAIN_SERIAL_MICRO rows={rows} cols={cols} jobs=1 cold_cache={cold} sample={sample} baseline_ns={} serial_ns={} conditioning_bytes=268435456 exact=true\n",times[0],times[1]));
            }}
        }
        receipt.push_str(&format!("PLAIN SERIAL PASS rows={rows} cols={cols} jobs=1,2,8 repeats=3 raw_bits=true\n"));
    }Ok(())
}
