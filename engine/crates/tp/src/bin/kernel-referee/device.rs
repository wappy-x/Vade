use std::{fs,path::{Path,PathBuf},time::{Duration,Instant}};
use engine_model::{plan::{ConfigPlan,admission::CheckpointAdmission,storage::WeightStorage,layer::Topology},safetensors::{read_bounded,Dtype},shapes::{CONFIG_BYTES,HEADER_BYTES}};
use spark_gpu_probe::{driver::{Backend,DeviceVector,RefereeShape,ResidentProjection,with_backend},expert_ops::{Shape,Op}};
#[path="../../../../../tools/export/src/sha256.rs"] mod sha256;
type Result<T>=std::result::Result<T,String>;
struct Matrix<'a> {shape:RefereeShape,input:DeviceVector<'a>,weight:DeviceVector<'a>,scales:Option<DeviceVector<'a>>}
struct Family<'a> {shape:RefereeShape,matrices:Vec<Matrix<'a>>}
fn bytes(values:&[f32])->Vec<u8> {values.iter().flat_map(|value|value.to_le_bytes()).collect()}
fn input(cols:usize)->Vec<f32> {(0..cols).map(|index|if index%37==0 {-0.0} else {((index*13+index/7)%61) as f32/32.0-0.9375}).collect()}
fn raw<'a>(backend:&Backend<'a>,values:&[u8])->Result<DeviceVector<'a>> {let mut vector=backend.vector(values.len().div_ceil(4))?;backend.write_vector_bytes(&mut vector,values)?;backend.synchronize()?;Ok(vector)}
fn expert(shape:RefereeShape)->Shape {Shape {op:if shape.format==2 {Op::NvRows} else {Op::Fp8Rows},rows:shape.rows,cols:shape.cols,block_rows:shape.block_rows,block_cols:shape.block_cols,tokens:1,global:shape.global,activation:shape.activation,bound:0.0,role:0}}
fn load<'a>(backend:&Backend<'a>,checkpoint:&CheckpointAdmission,out:&Path)->Result<Vec<Family<'a>>> {
    let topology=Topology {rank:0,parts:2};let mut families=Vec::new();
    for format in [0,1,2] {
        let mut names=Vec::new();let mut extent=None;
        for view in checkpoint.headers().weights().values() {
            let logical=view.logical();if logical.shape.len()!=2 || !matches!(view.binding().role.place,engine_model::shapes::Place::Layer(_)) {continue;}
            let matching=match (format,logical.storage) {(0,WeightStorage::Fp8Block {..})|(2,WeightStorage::Nvfp4 {..})=>true,(1,WeightStorage::Plain)=>matches!(view.components()[0].dtype,Dtype::F32|Dtype::Bf16) && !matches!(view.binding().operation,engine_model::plan::glm_index::TensorRole::Mhc {..}|engine_model::plan::glm_index::TensorRole::Embedding|engine_model::plan::glm_index::TensorRole::Head),_=>false};if !matching {continue;}
            let rows=spark_tp::resident_weights::Rows::new(&logical.shape,logical.storage,topology)?;let cols=logical.shape[1] as usize;
            if format!=1 && (rows.count!=1024 || cols!=4096) {continue;}
            if format==1 && (rows.count!=16384 || cols!=512) {continue;}
            if extent.is_some_and(|value|value!=(rows.count,cols)) {continue;}extent=Some((rows.count,cols));names.push(logical.name.clone());if names.len()==8 {break;}
        }
        if names.len()!=8 {return Err(format!("need eight distinct real matrices format={format}; found={}",names.len()));}
        let (rows,cols)=extent.unwrap();let mut shape=RefereeShape {format,rows:rows as u32,cols:cols as u32,block_rows:1,block_cols:16,global:1.0,activation:1.0};let mut matrices=Vec::new();
        for (job,name) in names.into_iter().enumerate() {
            let upload=checkpoint.upload(&name).map_err(|error|error.to_string())?;let selected=spark_tp::resident_weights::Rows::new(&upload.shape,upload.storage,topology)?;
            let (weight,scales)=match upload.storage {
                WeightStorage::Plain=>{let width=upload.dtype.bytes() as usize;let source=&upload.weight[selected.start*cols*width..(selected.start+rows)*cols*width];let values=match upload.dtype {Dtype::F32=>source.to_vec(),Dtype::Bf16=>source.chunks_exact(2).flat_map(|word|(u32::from(u16::from_le_bytes(word.try_into().unwrap()))<<16).to_le_bytes()).collect(),_=>return Err("plain dtype differs".into())};(values,Vec::new())},
                WeightStorage::Fp8Block {block}=>{shape.block_rows=block[0] as u32;shape.block_cols=block[1] as u32;let stride=cols.div_ceil(block[1] as usize)*4;let first=selected.start/block[0] as usize*stride;let length=rows.div_ceil(block[0] as usize)*stride;(upload.weight[selected.start*cols..(selected.start+rows)*cols].to_vec(),upload.scales[first..first+length].to_vec())},
                WeightStorage::Nvfp4 {..}=>{shape.global=upload.global_scale.ok_or("missing global calibration")?;shape.activation=upload.input_scale.ok_or("missing activation calibration")?;(upload.weight[selected.start*cols/2..(selected.start+rows)*cols/2].to_vec(),upload.scales[selected.start*cols/16..(selected.start+rows)*cols/16].to_vec())},_=>return Err("storage differs".into()),
            };
            let source=backend.upload_vector(&input(cols))?;let source=if format==2 {backend.expert_operation(Shape {op:Op::QuantizeRows,..expert(shape)},[Some(&source),None,None,None],None,None)?} else {source};
            let prefix=format!("f{format}-j{job}");fs::write(out.join(format!("{prefix}.weight")),&weight).map_err(|error|error.to_string())?;fs::write(out.join(format!("{prefix}.scales")),&scales).map_err(|error|error.to_string())?;fs::write(out.join(format!("{prefix}.input")),bytes(&backend.read_vector_values(&source)?)).map_err(|error|error.to_string())?;
            fs::write(out.join(format!("{prefix}.shape")),format!("{} {} {} {} {} {} {}\n",format,rows,cols,shape.block_rows,shape.block_cols,shape.global.to_bits(),shape.activation.to_bits())).map_err(|error|error.to_string())?;
            println!("MATRIX format={format} job={job} rows={rows} cols={cols} tensor={name} weight_sha256={} scales_sha256={}",sha256::digest(&weight),sha256::digest(&scales));
            matrices.push(Matrix {shape,input:source,weight:raw(backend,&weight)?,scales:if scales.is_empty() {None} else {Some(raw(backend,&scales)?)} });
        }
        fs::write(out.join(format!("f{format}.shape")),format!("{} {} {} {} {} {} {}\n",format,rows,cols,shape.block_rows,shape.block_cols,shape.global.to_bits(),shape.activation.to_bits())).map_err(|error|error.to_string())?;
        families.push(Family {shape,matrices});
    }Ok(families)
}
fn median(values:&mut [u128])->u128 {values.sort_unstable();values[values.len()/2]}
fn compare(expected:&[u8],actual:&[u8])->Result<()> {if expected.len()!=actual.len() {return Err("output lengths differ".into());}if let Some(index)=expected.iter().zip(actual).position(|(left,right)|left!=right) {return Err(format!("first_byte={index} expected={} actual={}",expected[index],actual[index]));}Ok(())}
pub(super) fn run()->Result<()> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();if args.len()!=5 {return Err("usage: kernel-referee CHECKPOINT BASELINE_CUBIN CANDIDATE_CUBIN ENGINEER OUT".into());}
    let deadline=Instant::now()+Duration::from_secs(55);let root=Path::new(&args[0]);let out=PathBuf::from(&args[4]);fs::create_dir(&out).map_err(|error|error.to_string())?;
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|error|error.to_string())?).map_err(|error|error.to_string())? else {return Err("deferred checkpoint required".into());};
    let checkpoint=description.open_headers(root,&read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|error|error.to_string())?).map_err(|error|error.to_string())?;
    let baseline=fs::read(&args[1]).map_err(|error|error.to_string())?;let candidate=fs::read(&args[2]).map_err(|error|error.to_string())?;let engineer=&args[3];
    with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
        backend.fixture_replace_baseline(&baseline)?;backend.configure_vector_pool(512<<20);let families=load(backend,&checkpoint,&out)?;let lut=backend.upload_vector(&spark_gpu_probe::expert_ops::decode_lut())?;
        let pressure=backend.upload_vector(&vec![0.5;64*1024*1024])?;let mut swept=backend.vector(pressure.len())?;let pressure_shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::StreamMean,heads:1,dim:pressure.len() as u32,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
        backend.fixture_replace_candidate(&candidate)?;backend.set_profiling(true);
        println!("IDENTITY engineer={engineer} baseline_sha256={} candidate_sha256={} pressure_bytes=268435456 pressure=baseline_StreamMean_read_write_before_every_sample repetitions=5 input=deterministic_finite_real_weights=true",sha256::digest(&baseline),sha256::digest(&candidate));
        for family in &families {for count in [1usize,2,8] {for new_order in [false,true] {
            if Instant::now()>=deadline {return Err("55-second session budget exhausted".into());}
            let shape=family.shape;let jobs=&family.matrices[..count];backend.select_arm(true)?;
            let borrowed=jobs.iter().map(|matrix|(&matrix.input,&matrix.weight,matrix.scales.as_ref())).collect::<Vec<_>>();
            let calibrations=jobs.iter().map(|matrix|(matrix.shape.global,matrix.shape.activation)).collect::<Vec<_>>();
            let plan=match backend.fixture_referee_plan(engineer,new_order,shape,&borrowed,&calibrations) {Ok(plan)=>plan,Err(error)=>{println!("UNAVAILABLE format={} count={count} track={} reason={error}",shape.format,if new_order {"B"} else {"A"});continue;}};
            let mut reference:Option<Vec<u8>>=None;let mut candidate_reference:Option<Vec<u8>>=None;let mut timings=[Vec::new(),Vec::new()];
            for repeat in 0..5 {for candidate_arm in if repeat%2==0 {[false,true]} else {[true,false]} {
                backend.select_arm(false)?;backend.resident_causal_into(pressure_shape,[Some(&pressure),None,None,None],None,&mut swept)?;backend.synchronize()?;
                backend.select_arm(candidate_arm)?;let before=backend.stats();
                let baseline_output;
                let output=if candidate_arm {backend.fixture_referee_launch(&plan)?;plan.output()} else {
                    baseline_output=if shape.format==1 {let mut output=backend.vector(shape.rows as usize*count)?;let parts=jobs.iter().map(|matrix|ResidentProjection::Plain {rows:shape.rows as usize,cols:shape.cols as usize,weight:&matrix.weight,input:&matrix.input}).collect::<Vec<_>>();backend.resident_projection_batch(&parts,&mut output,false)?;output} else if shape.format==2 {let parts=jobs.iter().map(|matrix|(Shape {op:Op::NvProject,..expert(matrix.shape)},&matrix.input,&matrix.weight,matrix.scales.as_ref().unwrap())).collect::<Vec<_>>();backend.expert_projection_batch(&parts)?} else {let parts=jobs.iter().map(|matrix|(expert(matrix.shape),Some((&matrix.weight,matrix.scales.as_ref().unwrap())),0)).collect::<Vec<_>>();backend.expert_group(&parts,&jobs[0].input,jobs[0].input.len(),&lut)?};&baseline_output
                };backend.synchronize()?;let stats=backend.stats().since(before);let ns=stats.device_kernel_ns.iter().sum::<u128>();timings[candidate_arm as usize].push(ns);let actual=bytes(&backend.read_vector_values(output)?);
                let saved=if candidate_arm {&mut candidate_reference} else {&mut reference};if let Some(expected)=saved {compare(expected,&actual)?;} else {*saved=Some(actual.clone());}
                fs::write(out.join(format!("f{}-n{count}-{}-r{repeat}-{}.f32",shape.format,if new_order {"B"} else {"A"},if candidate_arm {"candidate"} else {"baseline"})),&actual).map_err(|error|error.to_string())?;
                println!("SAMPLE format={} count={count} track={} repeat={repeat} arm={} ns={ns} launches={}",shape.format,if new_order {"B"} else {"A"},if candidate_arm {"candidate"} else {"baseline"},stats.queued_calls);
            }}
            let baseline_ns=median(&mut timings[0]);let candidate_ns=median(&mut timings[1]);let correct=compare(reference.as_ref().unwrap(),candidate_reference.as_ref().unwrap());
            println!("RESULT engineer={engineer} format={} rows={} cols={} count={count} track={} baseline_ns={baseline_ns} candidate_ns={candidate_ns} mac_per_second={:.3} speedup={:.9} baseline_bytes={} host_oracle=PENDING",shape.format,shape.rows,shape.cols,if new_order {"B"} else {"A"},shape.rows as f64*shape.cols as f64*count as f64*1e9/candidate_ns as f64,baseline_ns as f64/candidate_ns as f64,if correct.is_ok() {"EQUAL"} else {"DIFFERENT"});
            if !new_order {correct?;}
        }}}
        backend.synchronize()?;Ok(())
    })?;fs::write(out.join("COMPLETE"),b"GPU timing and repeat checks completed; B host oracle is separate\n").map_err(|error|error.to_string())?;println!("REFEREE GPU COMPLETE remaining_seconds={:.3}",deadline.saturating_duration_since(Instant::now()).as_secs_f64());Ok(())
}
