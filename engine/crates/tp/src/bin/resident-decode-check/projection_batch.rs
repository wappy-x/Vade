use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{driver::{with_backend,ResidentProjection,NormOutput},mxfp8_projection::Shape,norm_boundary::WeightType,causal_ops::{Shape as CausalShape,Op}};
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|value|value.to_bits()).collect()}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        for (case,(cols,mx_rows,plain_rows)) in [(32,[1,5,9],[1,7,129]),(256,[17,33,65],[65,32,17]),(4096,[4096,4096,4096],[64,32,64])].into_iter().enumerate() {
            let source=(0..cols).map(|feature|match feature%11 {0=>-0.0,1=>0.0,2=>16_777_216.0,3=>-16_777_216.0,_=>(feature as i32%31-15) as f32/8.0}).collect::<Vec<_>>();
            let decoded=spark_gpu_probe::dynamic_mxfp8::Activation::from_f32_positive_zero(&source)?;
            let source=backend.upload_vector(&source)?;let decoded=NormOutput::F32(backend.upload_vector(decoded.decoded())?);
            let mut matrices=Vec::new();let mut plain=Vec::new();
            for job in 0..3 {
                let shape=Shape::new(mx_rows[job],cols,WeightType::F32)?;
                let mut bytes=(0..shape.weight_bytes()).map(|index|{let magnitude=0x20+((index*7+job*3)%79) as u8;magnitude|if index%5==0 {128} else {0}}).collect::<Vec<_>>();
                bytes.extend((0..shape.scale_bytes()).map(|index|120+((index+job)%12) as u8));
                let path=out.join(format!("case{case}-mx{job}.raw"));super::write(&path,&bytes)?;let file=fs::File::open(&path).map_err(|error|error.to_string())?;
                let ranges=[spark_gpu_probe::load::WeightRange {file_offset:0,destination:0,bytes:bytes.len(),row_bytes:0,row_stride:0}];
                let arena=backend.load_weight_arena(bytes.len(),&[spark_gpu_probe::load::WeightFile {file:&file,ranges:&ranges}])?;
                matrices.push(backend.resident_mxfp8_projection(&arena,shape)?);
                plain.push(backend.upload_vector(&(0..plain_rows[job]*cols).map(|index|(index as i32%23-11) as f32/32.0).collect::<Vec<_>>())?);
            }
            for round in [false,true] {
                let mut expected=Vec::new();
                for matrix in &matrices {let output=matrix.apply(&decoded)?;expected.push(backend.read_vector_values(output.output().vector())?);}
                for job in 0..3 {let shape=CausalShape {op:Op::Project,heads:0,dim:0,taps:0,rows:plain_rows[job] as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
                    let output=backend.causal_operation(shape,[Some(&source),None,None,None],Some(&plain[job]))?;
                    expected.push(if round {backend.read_vector_values(backend.round_bf16_lanes(&output)?.vector())?} else {backend.read_vector_values(&output)?});
                }
                for order in [&[0usize,3,1,4,2,5][..],&[2usize,0,3,5,1,4,0,3,2][..]] {
                    let jobs=order.iter().map(|&job|if job<3 {ResidentProjection::Mxfp8 {matrix:&matrices[job],input:decoded.vector()}}
                        else {let job=job-3;ResidentProjection::Plain {rows:plain_rows[job],cols,weight:&plain[job],input:&source}}).collect::<Vec<_>>();
                    let expected=order.iter().flat_map(|&job|expected[job].iter().copied()).collect::<Vec<_>>();
                    let mut output=backend.vector(expected.len())?;
                    for queued in [false,true] {let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
                    for repeat in 1..=3 {let before=backend.stats();backend.resident_projection_batch(&jobs,&mut output,round)?;
                        if queued && backend.stats().since(before).stream_wait_calls!=0 {return Err("queued mixed projection waited before its consumer".into());}
                        let values=backend.read_vector_values(&output)?;
                        if bits(&values)!=bits(&expected) {return Err(format!("mixed projection bits differ case={case} round={round} jobs={} repeat={repeat}",jobs.len()));}
                        super::write(&out.join(format!("case{case}-round{round}-queue{queued}-n{}-{repeat}.f32",jobs.len())),&super::floats(&values)?)?;
                        receipt.push_str(&format!("MIXED PROJECTION DEVICE PASS case={case} cols={cols} round={round} jobs={} repeat={repeat} complete_logits=true repeated_jobs=true queued={queued}\n",jobs.len()));
                    }
                    drop(queue);}
                    let mut short=backend.vector(expected.len()-1)?;if backend.resident_projection_batch(&jobs,&mut short,round).is_ok() {return Err("short mixed output admitted".into());}
                }
            }
            let mut output=backend.vector(1)?;if backend.resident_projection_batch(&[],&mut output,true).is_ok() {return Err("empty mixed batch admitted".into());}
            let short=backend.upload_vector(&[0.0])?;
            if backend.resident_projection_batch(&[ResidentProjection::Mxfp8 {matrix:&matrices[0],input:&short}],&mut output,true).is_ok() {return Err("short mixed input admitted".into());}
            if backend.resident_projection_batch(&[ResidentProjection::Plain {rows:1,cols,weight:&short,input:&source}],&mut output,true).is_ok() {return Err("short mixed weight admitted".into());}
        }
        for repeat in 0..3 {
            let queue=backend.resident_queue()?;backend.enable_resident_launches()?;let before=backend.stats();
            let mut output=backend.vector(3)?;
            {
                let source=backend.upload_vector(&vec![1.0;32])?;let weight=backend.upload_vector(&(0..96).map(|i|if i/32==1 {-0.0} else {1.0}).collect::<Vec<_>>())?;
                backend.resident_projection_batch(&[ResidentProjection::Plain {rows:3,cols:32,weight:&weight,input:&source}],&mut output,true)?;
            }
            if backend.stats().since(before).stream_wait_calls!=0 {return Err("mixed operand drop drained the owned queue".into());}
            let got=backend.read_vector_values(&output)?;if bits(&got)!=bits(&[32.0,0.0,32.0]) {return Err("queued mixed early owner drop changed bits".into());}
            if backend.stats().since(before).graph_launches==0 {return Err("queued mixed batch did not replay".into());}
            receipt.push_str(&format!("MIXED PROJECTION QUEUED PASS repeat={repeat} early_owner_drop=true graph_replay=true no_early_wait=true\n"));drop(queue);
        }
        Ok(())
    })?;
    super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
