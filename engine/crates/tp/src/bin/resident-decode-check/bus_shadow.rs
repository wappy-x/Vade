use super::*;
use spark_gpu_probe::{driver::{with_backend,Backend,DeviceVector,NormOutput},expert_ops::{Op,Shape},mxfp8_projection,norm_boundary::WeightType};
fn raw<'d>(backend:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {
    let mut bytes=bytes.to_vec();bytes.resize(bytes.len().div_ceil(4)*4,0);
    let mut value=backend.vector(bytes.len()/4)?;backend.write_vector_bytes(&mut value,&bytes)?;value.into_resident_shared()
}
fn equal(actual:&[f32],expected:&[f32])->Result<()> {
    if actual.len()!=expected.len() || actual.iter().zip(expected).any(|(actual,expected)|actual.to_bits()!=expected.to_bits()) {return Err("bus shadow output bytes differ".into());}Ok(())
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut receipt=String::new();let mut compared=0usize;
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        for (case,(rows,cols)) in [(33usize,512usize),(2048,2048),(4097,4096)].into_iter().enumerate() {
            let shape=mxfp8_projection::Shape::new(rows,cols,WeightType::F32)?;
            let mut weights=(0..shape.weight_bytes()).map(|at|0x20+(at%47) as u8).collect::<Vec<_>>();weights.extend(vec![127;shape.scale_bytes()]);
            let path=out.join(format!("mx-{case}.raw"));super::write(&path,&weights)?;let file=fs::File::open(path).map_err(|error|error.to_string())?;
            let ranges=[spark_gpu_probe::load::WeightRange {file_offset:0,destination:0,bytes:weights.len(),row_bytes:0,row_stride:0}];
            let arena=backend.load_weight_arena(weights.len(),&[spark_gpu_probe::load::WeightFile {file:&file,ranges:&ranges}])?;
            let matrix=backend.resident_mxfp8_projection(&arena,shape)?;
            let values=(0..cols).map(|at|(at as i32%31-15) as f32/16.0).collect::<Vec<_>>();
            let input=NormOutput::F32(backend.upload_vector(&values)?);let pass=matrix.apply(&input)?;
            let expected=backend.read_vector_values(pass.output().vector())?;
            for row_count in [1,2,8] {
                let mailbox=backend.tp_mailbox(cols*4)?;let mut host=mailbox.take_host().ok_or("missing mailbox host")?;
                let guard=f32::from_bits(0x3f123456);let mut target=backend.upload_vector(&vec![guard;cols+32])?.into_resident_shared()?;
                let input=NormOutput::F32(target.resident_range(16..16+cols)?);
                let segments=(0..row_count).map(|row|[row*cols/row_count,16+row*cols/row_count,cols/row_count]).collect::<Vec<_>>();
                let before=spark_gpu_probe::bus_shadow::counts();let queue=backend.resident_queue()?;backend.enable_resident_launches()?;
                for repeat in 1..=3 {
                    host.deliver(repeat,&super::floats(&values)?)?;
                    backend.tp_mailbox_take_scatter(&mailbox,&mut target,&segments,0,repeat,1_000_000_000)?;
                    let pass=matrix.apply(&input)?;let actual=backend.read_vector_values(pass.output().vector())?;equal(&actual,&expected)?;
                    let actual_target=backend.read_vector_values(&target)?;let mut expected_target=vec![guard;cols+32];expected_target[16..16+cols].copy_from_slice(&values);equal(&actual_target,&expected_target)?;
                    mailbox.require_success()?;compared+=(actual.len()+actual_target.len())*4;
                    super::write(&out.join(format!("mx-{case}-{row_count}-{repeat}.f32")),&super::floats(&actual)?)?;
                }
                drop(queue);let after=spark_gpu_probe::bus_shadow::counts();if after[0]<=before[0] {return Err("static shadow fixture did not arm".into());}
                receipt.push_str(&format!("MX PASS rows={rows} cols={cols} receive_rows={row_count} repeats=3 armed={}\n",after[0]-before[0]));
            }
        }
        Ok(())
    })?;
    for (rows,cols) in [(33u32,512u32),(2048,2048)] {
        for pattern in [vec![0,1,2,3,4,5,6,7],vec![15,13,11,9,7,5,3,1],vec![7,0,6,1,5,2,4,3]] {
            with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
            // TENSOR-CORE REBASE: only a column-tile table is a device route; a shape outside the storage rule is skipped.
            if !spark_gpu_probe::expert_ops::nv_tile_layout_admits(rows as usize,cols as usize) {return Ok(());}
            let shape=Shape {op:Op::NvProject,rows,cols,block_rows:spark_gpu_probe::expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:1.0,activation:0.25,bound:0.0,role:0};
            let owners=(0..16).map(|expert|Ok((raw(backend,&vec![0x22+expert as u8;rows as usize*cols as usize/2])?,raw(backend,&vec![0x38;rows as usize*cols as usize/16])?))).collect::<Result<Vec<_>>>()?;
            let parts=owners.iter().map(|(weight,scales)|(shape,weight,scales)).collect::<Vec<_>>();let table=backend.device_route_table(&parts,16,8,1,true)?.ok_or("route not admitted")?;
            let words=cols as usize*8;let values=(0..words).map(|at|(at as i32%23-11) as f32/8.0).collect::<Vec<_>>();let source=backend.upload_vector(&values)?.into_resident_shared()?;
                let ids=backend.upload_vector(&pattern.iter().map(|&id|id as f32).collect::<Vec<_>>())?;
                let expected=backend.device_route_projection(&table,&ids,&source)?;let expected=backend.read_vector_values(&expected)?;
                let mailbox=backend.tp_mailbox(words*4)?;let mut host=mailbox.take_host().ok_or("missing mailbox host")?;
                let mut target=backend.upload_vector(&vec![-1.0;words])?.into_resident_shared()?;
                let segments=(0..8).map(|slot|[slot*cols as usize,slot*cols as usize,cols as usize]).collect::<Vec<_>>();
                let before=spark_gpu_probe::bus_shadow::counts();let queue=backend.resident_queue()?;backend.enable_resident_launches()?;
                for repeat in 1..=3 {
                    host.deliver(repeat,&super::floats(&values)?)?;
                    let prior=backend.device_route_projection(&table,&ids,&source)?;
                    backend.tp_mailbox_take_scatter(&mailbox,&mut target,&segments,0,repeat,1_000_000_000)?;
                    let actual=backend.device_route_projection(&table,&ids,&target)?;
                    let (prior,actual)=backend.read_resident_pair(&prior,&actual)?;equal(&prior,&expected)?;equal(&actual,&expected)?;
                    equal(&backend.read_vector_values(&target)?,&values)?;mailbox.require_success()?;compared+=(prior.len()+actual.len()+values.len())*4;
                    super::write(&out.join(format!("nv-{rows}-{}-{repeat}.f32",pattern[0])),&super::floats(&actual)?)?;
                }
                drop(queue);let after=spark_gpu_probe::bus_shadow::counts();if after[1]<=before[1] {return Err("routed shadow fixture did not arm".into());}
                receipt.push_str(&format!("NV PASS rows={rows} cols={cols} ids={pattern:?} repeats=3 armed={}\n",after[1]-before[1]));
                Ok(())
            })?;
        }
    }
    receipt.push_str(&format!("BUS SHADOW COMPLETE compared_bytes={compared} differing_bytes=0\n"));super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
