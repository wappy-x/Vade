use super::*;
use std::{cell::RefCell,io::{self,BufRead,Write},rc::Rc,sync::mpsc};
use spark_gpu_probe::{driver::{Backend,DeviceVector,ResidentMhcBoundary,ResidentProjection,Shadow,with_backend},expert_ops::{Shape,Op},norm_boundary::WeightType};
use spark_tp::{dynamic_projection::ResidentDynamicProjection,mxfp8_projection::Mxfp8Projection};

enum Storage<'b,'d> {
    Packed {shape:Shape,weight:DeviceVector<'d>,scales:DeviceVector<'d>,input:DeviceVector<'d>},
    Mx {matrix:ResidentDynamicProjection<'b,'d>,input:DeviceVector<'d>},
    Plain {shape:spark_gpu_probe::causal_ops::Shape,weight:DeviceVector<'d>,input:DeviceVector<'d>},
    Mhc {matrix:ResidentMhcBoundary<'b,'d>,input:Rc<DeviceVector<'d>>,block:Rc<DeviceVector<'d>>},
}
struct Case<'b,'d> {name:String,macs:u128,tokens:u32,scalar:bool,storage:Rc<Storage<'b,'d>>}
struct Sample {kernel_ns:u128,wall_ns:u128,launches:u128,planes:Vec<Vec<u8>>}
struct CachePressure<'d> {source:DeviceVector<'d>,output:RefCell<DeviceVector<'d>>,shape:spark_gpu_probe::causal_ops::Shape}
struct Bench<'b,'d> {backend:&'b Backend<'d>,cases:Vec<Case<'b,'d>>,reference:Vec<Vec<Vec<u8>>>,lut:DeviceVector<'d>,deadline:Instant,out:&'b Path,pressure:Option<CachePressure<'d>>,
    checkpoint:&'b CheckpointAdmission,parts:u32,shadow:RefCell<Option<Rc<ShadowCase<'b,'d>>>>}
/// CHAIN-SHADOW screen: one real after-site boundary, then the MXFP8 projections that read its output.
struct ShadowCase<'b,'d> {matrix:ResidentMhcBoundary<'b,'d>,input:Rc<DeviceVector<'d>>,block:Rc<DeviceVector<'d>>,
    projections:Vec<(String,spark_tp::dynamic_projection::ResidentDynamicProjection<'b,'d>)>,decoded:DeviceVector<'d>,rows:usize,reference:RefCell<Option<Vec<Vec<u8>>>>}
struct ShadowSample {boundary_ns:u128,project_ns:u128,total_ns:u128,planes:Vec<Vec<u8>>}
fn load_shadow<'b,'d>(backend:&'b Backend<'d>,checkpoint:&CheckpointAdmission,parts:u32)->Result<ShadowCase<'b,'d>> {
    use engine_model::plan::{state::StatePlan,layer::Kind};
    let topology=Topology {rank:0,parts};let started=Instant::now();
    let state=StatePlan::new(checkpoint.description(),topology,1,isize::MAX as usize).map_err(|error|error.to_string())?;
    let boundary=spark_tp::layer_driver::LayerBoundary::new(checkpoint.description(),checkpoint.headers(),&state,0,1,topology,usize::MAX)?;
    let steps=&boundary.program().steps;
    let step=steps.iter().position(|step|step.kind==Kind::MhcPrepare).ok_or("no mHC boundary in first layer")?;
    let operands=boundary.prepare_residual_block_checkpoint(step,checkpoint)?;let shape=operands.mhc().shape();
    let matrix=operands.upload_boundary(backend)?;
    // The MXFP8 projections after this boundary that share the first one's input stream next.
    let later=steps[step+1..].iter().take_while(|s|s.kind!=Kind::MhcPrepare).filter(|s|s.kind==Kind::Project && s.weights.len()==1 && matches!(s.weights[0].storage,WeightStorage::Mxfp8)).collect::<Vec<_>>();
    let first=later.first().ok_or("no MXFP8 projection follows the first boundary")?;
    let names=later.iter().filter(|s|s.inputs==first.inputs).take(3).map(|s|s.weights[0].name.clone()).collect::<Vec<_>>();
    let mut projections=Vec::new();
    for name in names {
        let matrix=spark_tp::mxfp8_projection::Mxfp8Projection::matrix_rows(checkpoint,&name,WeightType::F32,topology)?.load_dynamic_positive_zero(backend)?;
        projections.push((name,matrix));
    }
    let cols=shape.hidden;let source=backend.upload_vector(&input(cols as usize))?;
    let quantize=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Mxfp8QuantizePositiveZero,heads:0,dim:0,taps:0,rows:0,cols,epsilon:0.0,bound:0.0};
    let decoded=backend.causal_operation(quantize,[Some(&source),None,None,None],None)?;
    let rows=projections.iter().map(|(_,matrix)|matrix.resident().rows()).sum();
    let site=Rc::new(backend.upload_vector(&input(shape.wide()))?);let block=Rc::new(backend.upload_vector(&input(shape.hidden as usize))?);
    backend.synchronize()?;
    for (name,matrix) in &projections {println!("SHADOW LOAD tensor={name} rows={} e4m3_bytes={}",matrix.resident().rows(),matrix.resident().rows()*cols as usize);}
    println!("SHADOW LOAD PASS boundary_rows={} streams={} hidden={} projections={} output_rows={rows} seconds={:.6}",shape.rows,shape.streams,shape.hidden,projections.len(),started.elapsed().as_secs_f64());
    Ok(ShadowCase {matrix,input:site,block,projections,decoded,rows,reference:RefCell::new(None)})
}
fn check_time(deadline:Instant)->Result<()> {if Instant::now()>=deadline {Err("session budget exhausted; no new launch was submitted".into())} else {Ok(())}}
fn raw<'d>(backend:&Backend<'d>,values:&[u8])->Result<DeviceVector<'d>> {
    let mut output=backend.vector(values.len().div_ceil(4))?;backend.write_vector_bytes(&mut output,values)?;backend.synchronize()?;Ok(output)
}
fn input(cols:usize)->Vec<f32> {(0..cols).map(|index|if index%37==0 {-0.0} else {((index*13+index/7)%61) as f32/32.0-0.9375}).collect()}
fn load<'b,'d>(backend:&'b Backend<'d>,checkpoint:&CheckpointAdmission,selections:&[Selection],parts:u32,deadline:Instant,packed_tokens:u32)->Result<Vec<Case<'b,'d>>> {
    let topology=Topology {rank:0,parts};let mut cases=Vec::new();
    for (index,selection) in selections.iter().enumerate() {
        check_time(deadline)?;let started=Instant::now();let (tokens,input_values,output_values)=token_layout(selection,packed_tokens)?;
        let source=backend.upload_vector(&input(input_values))?;
        let storage=if selection.kind=="mxfp8" {
            let matrix=Mxfp8Projection::matrix_rows(checkpoint,&selection.name,WeightType::F32,topology)?.load_dynamic_positive_zero(backend)?;
            let shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Mxfp8QuantizePositiveZero,heads:0,dim:0,taps:0,rows:0,cols:selection.cols as u32,epsilon:0.0,bound:0.0};
            let decoded=backend.causal_operation(shape,[Some(&source),None,None,None],None)?;
            Storage::Mx {matrix,input:decoded}
        } else if selection.kind=="plain-fp32" {
            let upload=checkpoint.upload(&selection.name).map_err(|error|error.to_string())?;
            let rows=spark_tp::resident_weights::Rows::new(&upload.shape,upload.storage,topology)?;
            let width=upload.dtype.bytes() as usize;let selected=upload.weight.get(rows.start*selection.cols*width..(rows.start+rows.count)*selection.cols*width).ok_or("plain matrix selection exceeds payload")?;
            let values=match upload.dtype {
                engine_model::safetensors::Dtype::F32=>selected.chunks_exact(4).map(|value|f32::from_le_bytes(value.try_into().unwrap())).collect::<Vec<_>>(),
                engine_model::safetensors::Dtype::Bf16=>selected.chunks_exact(2).map(|value|f32::from_bits(u32::from(u16::from_le_bytes(value.try_into().unwrap()))<<16)).collect(),
                _=>return Err("plain matrix source precision differs".into()),
            };
            let shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:rows.count as u32,cols:selection.cols as u32,epsilon:0.0,bound:0.0};
            Storage::Plain {shape,weight:backend.upload_vector(&values)?,input:source}
        } else {
            let upload=checkpoint.upload(&selection.name).map_err(|error|error.to_string())?;
            let rows=spark_tp::resident_weights::Rows::new(&upload.shape,upload.storage,topology)?;
            let mut shape=Shape {op:Op::NvRows,rows:selection.rows as u32,cols:selection.cols as u32,tokens,block_rows:1,block_cols:16,
                global:upload.global_scale.unwrap_or(1.0),activation:upload.input_scale.unwrap_or(1.0),bound:0.0};
            let (stride,scale_start,scale_len)=match upload.storage {
                WeightStorage::Nvfp4 {..}=>{
                    if upload.global_scale.is_none() || upload.input_scale.is_none() {return Err("NVFP4 calibration scalars are absent".into());}
                    (selection.cols/2,rows.start*(selection.cols/16),selection.rows*(selection.cols/16))
                },
                WeightStorage::Fp8Block {block}=>{
                    shape.op=Op::Fp8Rows;shape.block_rows=block[0].try_into().map_err(|_|"scale block exceeds ABI")?;shape.block_cols=block[1].try_into().map_err(|_|"scale block exceeds ABI")?;
                    let grid=selection.cols.div_ceil(block[1] as usize);(selection.cols,rows.start/block[0] as usize*grid*4,selection.rows.div_ceil(block[0] as usize)*grid*4)
                },_=>return Err("selected projection storage differs".into()),
            };
            let weight=raw(backend,upload.weight.get(rows.start*stride..(rows.start+rows.count)*stride).ok_or("selected weight rows exceed payload")?)?;
            let scales=raw(backend,upload.scales.get(scale_start..scale_start+scale_len).ok_or("selected scales exceed payload")?)?;
            let source=if shape.op==Op::NvRows {backend.expert_operation(Shape {op:Op::QuantizeRows,..shape},[Some(&source),None,None,None],None,None)?} else {source};
            Storage::Packed {shape,weight,scales,input:source}
        };
        backend.synchronize()?;println!("LOAD PASS case=p{index} seconds={:.6} token_rows={tokens} matrices=1 input_values={input_values} output_values={output_values} tensor={}",started.elapsed().as_secs_f64(),selection.name);
        let storage=Rc::new(storage);let macs=selection.rows as u128*selection.cols as u128*u128::from(tokens);
        cases.push(Case {name:format!("p{index}"),macs,tokens,scalar:false,storage:storage.clone()});
        if tokens==1 {cases.push(Case {name:format!("p{index}-scalar"),macs,tokens,scalar:true,storage});}
    }
    check_time(deadline)?;
    use engine_model::plan::{state::StatePlan,layer::Kind};
    let state=StatePlan::new(checkpoint.description(),topology,1,isize::MAX as usize).map_err(|error|error.to_string())?;
    let boundary=spark_tp::layer_driver::LayerBoundary::new(checkpoint.description(),checkpoint.headers(),&state,0,1,topology,usize::MAX)?;
    let step=boundary.program().steps.iter().position(|step|step.kind==Kind::MhcPrepare).ok_or("no mHC boundary in first layer")?;
    let operands=boundary.prepare_residual_block_checkpoint(step,checkpoint)?;let shape=operands.mhc().shape();
    let matrix=operands.upload_boundary(backend)?;let source=Rc::new(backend.upload_vector(&input(shape.wide()))?);
    let block=Rc::new(backend.upload_vector(&input(shape.hidden as usize))?);
    println!("CASE id=mhc kind=real-checkpoint-boundary rows={} streams={} hidden={} iterations={} sites=3 macs_scope=coefficient-dots-only",shape.rows,shape.streams,shape.hidden,shape.iterations);
    cases.push(Case {name:"mhc".into(),macs:3*shape.wide() as u128*shape.coefficients() as u128,tokens:1,scalar:false,storage:Rc::new(Storage::Mhc {matrix,input:source,block})});Ok(cases)
}
impl<'b,'d> Bench<'b,'d> {
    fn sample(&self,index:usize,candidate:bool)->Result<Sample> {
        check_time(self.deadline)?;let backend=self.backend;backend.synchronize()?;backend.select_arm(false)?;
        if let Some(pressure)=&self.pressure {
            backend.resident_causal_into(pressure.shape,[Some(&pressure.source),None,None,None],None,&mut pressure.output.borrow_mut())?;backend.synchronize()?;
        }
        backend.select_arm(candidate)?;
        let case=&self.cases[index];let before=backend.stats();let started=Instant::now();
        let vectors=match &*case.storage {
            Storage::Packed {shape,weight,scales,input}=>vec![if case.scalar {
                backend.expert_operation(Shape {op:if shape.op==Op::NvRows {Op::NvProject} else {Op::Fp8Project},..*shape},[Some(input),None,None,None],Some(weight),Some(scales))?
            } else {backend.expert_group(&[(*shape,Some((weight,scales)),0)],input,input.len()/shape.tokens as usize,&self.lut)?}],
            Storage::Mx {matrix,input}=>{
                if case.scalar {vec![matrix.resident().apply_rows(input,1)?]} else {
                    let mut output=backend.vector(case.macs as usize/input.len())?;
                    backend.resident_projection_batch(&[ResidentProjection::Mxfp8 {matrix:matrix.resident(),input}],&mut output,false)?;vec![output]
                }
            },
            Storage::Plain {shape,weight,input}=>{
                if case.scalar {vec![backend.causal_operation(*shape,[Some(input),None,None,None],Some(weight))?]} else {
                    let mut output=backend.vector(shape.rows as usize)?;
                    backend.resident_projection_batch(&[ResidentProjection::Plain {rows:shape.rows as usize,cols:shape.cols as usize,weight,input}],&mut output,false)?;vec![output]
                }
            },
            Storage::Mhc {matrix,input,block}=>{
                let first=matrix.begin(input.clone())?;
                let second=matrix.begin_after(first,block.clone(),Rc::new(backend.vector(input.len())?))?;
                let third=matrix.begin_after(second,block.clone(),Rc::new(backend.vector(input.len())?))?;
                backend.synchronize()?;let stats=backend.stats().since(before);let wall_ns=started.elapsed().as_nanos();let mut planes=Vec::new();let mut pass=Some(&third);
                while let Some(site)=pass {for vector in [site.residual(),site.mixing_normalized(),site.coefficients(),site.collapsed(),&*site.normalized()] {
                    planes.push(bytes(&backend.read_vector_values(vector)?));
                }pass=site.previous();}
                return Ok(Sample {kernel_ns:stats.device_kernel_ns.iter().sum(),wall_ns,launches:stats.queued_calls,planes});
            },
        };
        backend.synchronize()?;let stats=backend.stats().since(before);let wall_ns=started.elapsed().as_nanos();
        let planes=vectors.iter().map(|vector|backend.read_vector_values(vector).map(|values|bytes(&values))).collect::<Result<Vec<_>>>()?;
        Ok(Sample {kernel_ns:stats.device_kernel_ns.iter().sum(),wall_ns,launches:stats.queued_calls,planes})
    }
    /// Dependent FP32 add latency: event time and SM cycles at four chain lengths.
    /// The slope between the two longest chains removes the fixed launch cost.
    fn add_latency(&self)->Result<()> {
        let backend=self.backend;let mut rows=Vec::new();let mut report=String::new();
        for adds in [0u32,16384,65536,262144] {for repeat in 0..5 {
            check_time(self.deadline)?;let (bits,cycles,timer_ns,event_ns)=backend.chain_add_latency(1.0,1.0/1024.0,adds)?;
            // The host replays the same serial chain; the device result must match it bit for bit.
            let expected=spark_gpu_probe::mhc_params::serial_add_bits(1.0,1.0/1024.0,adds);
            if bits!=expected {return Err(format!("ADDLAT BYTE FAIL adds={adds} device={bits:#x} host={expected:#x}"));}
            let line=format!("ADDLAT adds={adds} repeat={repeat} cycles={cycles} timer_ns={timer_ns} event_ns={event_ns} cycles_per_add={:.4} result_bits={bits:#010x} BYTE_PASS\n",if adds==0 {0.0} else {cycles as f64/f64::from(adds)});
            print!("{line}");report.push_str(&line);rows.push((adds,cycles,timer_ns,event_ns));
        }}
        let pick=|adds:u32,field:usize|->Result<u128> {let values=rows.iter().filter(|row|row.0==adds).map(|row|[row.1 as u128,row.2 as u128,row.3][field]).collect::<Vec<_>>();median(&values)};
        let (short,long)=(65536u32,262144u32);let span=f64::from(long-short);
        let cycles=(pick(long,0)? as f64-pick(short,0)? as f64)/span;let timer=(pick(long,1)? as f64-pick(short,1)? as f64)/span;let event=(pick(long,2)? as f64-pick(short,2)? as f64)/span;
        let clock=cycles/timer;
        let line=format!("ADDLAT RESULT cycles_per_add={cycles:.4} ns_per_add_timer={timer:.4} ns_per_add_event={event:.4} sm_ghz={clock:.4} chain16384_us={:.3} boundary_floor_us={:.3} step_floor_ms_90={:.3} scope=one-thread-dependent-fp32-add\n",
            event*16384.0/1000.0,2.0*event*16384.0/1000.0,90.0*2.0*event*16384.0/1e6);
        print!("{line}");report.push_str(&line);fs::write(self.out.join("addlat.txt"),report).map_err(|error|error.to_string())
    }
    fn shadow_case(&self)->Result<Rc<ShadowCase<'b,'d>>> {
        if let Some(case)=self.shadow.borrow().as_ref() {return Ok(case.clone());}
        check_time(self.deadline)?;let case=Rc::new(load_shadow(self.backend,self.checkpoint,self.parts)?);
        *self.shadow.borrow_mut()=Some(case.clone());Ok(case)
    }
    /// One sample: site 1, then site 2 (the production after-site form) with or
    /// without the shadow, then the projections. Three GPU marks bracket site 2
    /// and the projections. No host synchronize runs inside the timed span.
    fn shadow_sample(&self,case:&ShadowCase<'b,'d>,shadow:Option<&Shadow<'_>>)->Result<ShadowSample> {
        check_time(self.deadline)?;let backend=self.backend;backend.synchronize()?;backend.select_arm(false)?;
        if let Some(pressure)=&self.pressure {
            backend.resident_causal_into(pressure.shape,[Some(&pressure.source),None,None,None],None,&mut pressure.output.borrow_mut())?;backend.synchronize()?;
        }
        let mut output=backend.vector(case.rows)?;
        let (second,marks)={
            let queue=backend.resident_queue()?;
            let first=case.matrix.begin(case.input.clone())?;
            let start=backend.gpu_mark()?;
            let second=case.matrix.begin_after_shadowed(first,case.block.clone(),Rc::new(backend.vector(case.input.len())?),shadow)?;
            let middle=backend.gpu_mark()?;
            let jobs=case.projections.iter().map(|(_,matrix)|ResidentProjection::Mxfp8 {matrix:matrix.resident(),input:&case.decoded}).collect::<Vec<_>>();
            backend.resident_projection_batch(&jobs,&mut output,false)?;
            let end=backend.gpu_mark()?;drop(queue);(second,[start,middle,end])
        };
        backend.synchronize()?;
        let boundary_ns=marks[0].ns_until(&marks[1])?;let project_ns=marks[1].ns_until(&marks[2])?;let total_ns=marks[0].ns_until(&marks[2])?;
        let mut planes=Vec::new();let mut pass=Some(&second);
        while let Some(site)=pass {for vector in [site.residual(),site.mixing_normalized(),site.coefficients(),site.collapsed(),&*site.normalized()] {planes.push(bytes(&backend.read_vector_values(vector)?));}pass=site.previous();}
        planes.push(bytes(&backend.read_vector_values(&output)?));
        Ok(ShadowSample {boundary_ns,project_ns,total_ns,planes})
    }
    fn shadow_trial(&self,label:&str,dots_bytes:usize,finish_bytes:usize,blocks:u32,evict_last:bool,repeats:usize)->Result<()> {
        if repeats<3 || repeats>31 || repeats%2==0 {return Err("repeats must be odd and within 3..31".into());}
        let case=self.shadow_case()?;let (first,second)=(case.projections.first().ok_or("no shadow projection")?,case.projections.get(1));
        let dots=if dots_bytes==0 {None} else {Some(first.1.resident().shadow_range(0,dots_bytes)?)};
        let finish=if finish_bytes==0 {None} else {Some(second.ok_or("finish shadow needs a second projection")?.1.resident().shadow_range(0,finish_bytes)?)};
        let shadow=Shadow {dots,finish,blocks,evict_last};
        println!("SHADOW TRIAL label={label} dots_tensor={} dots_bytes={dots_bytes} finish_tensor={} finish_bytes={finish_bytes} tail_blocks={blocks} policy={} repeats={repeats}",
            first.0,second.map_or("none",|second|second.0.as_str()),if evict_last {"evict_last"} else {"normal"});
        let mut report=String::new();let mut timings=[Vec::new(),Vec::new()];
        let compare=|sample:&ShadowSample,arm:&str,repeat:usize|->Result<()> {
            let mut reference=case.reference.borrow_mut();
            let expected=reference.get_or_insert_with(||sample.planes.clone());
            if expected.len()!=sample.planes.len() {return Err("shadow plane count differs".into());}
            for (plane,(actual,reference)) in sample.planes.iter().zip(expected.iter()).enumerate() {
                fs::write(self.out.join(format!("{label}-{repeat}-{arm}-plane{plane}.f32")),actual).map_err(|error|error.to_string())?;
                if let Some(byte)=first_difference(reference,actual) {return Err(format!("SHADOW BYTE FAIL label={label} arm={arm} repeat={repeat} plane={plane} first_byte={byte}"));}
            }Ok(())
        };
        for arm in [false,true] {let sample=self.shadow_sample(&case,arm.then_some(&shadow))?;compare(&sample,if arm {"candidate"} else {"baseline"},0)?;}
        for repeat in 1..=repeats {for candidate in if repeat%2==1 {[false,true]} else {[true,false]} {
            let arm=if candidate {"candidate"} else {"baseline"};let sample=self.shadow_sample(&case,candidate.then_some(&shadow))?;compare(&sample,arm,repeat)?;
            let line=format!("SHADOW SAMPLE label={label} repeat={repeat} arm={arm} boundary_ns={} project_ns={} total_ns={} planes={} bytes={} PASS\n",sample.boundary_ns,sample.project_ns,sample.total_ns,sample.planes.len(),sample.planes.iter().map(Vec::len).sum::<usize>());
            print!("{line}");report.push_str(&line);timings[usize::from(candidate)].push([sample.boundary_ns,sample.project_ns,sample.total_ns]);
        }}
        let field=|arm:usize,index:usize|median(&timings[arm].iter().map(|row|row[index]).collect::<Vec<_>>());
        let (b0,b1,p0,p1,t0,t1)=(field(0,0)?,field(1,0)?,field(0,1)?,field(1,1)?,field(0,2)?,field(1,2)?);
        let saved=t0 as f64-t1 as f64;
        let line=format!("SHADOW RESULT label={label} boundary_ns_before={b0} boundary_ns_after={b1} project_ns_before={p0} project_ns_after={p1} total_ns_before={t0} total_ns_after={t1} saved_ns_per_boundary={saved:.0} saved_ms_per_step_x90={:.4} repeats={repeats} BYTE_PASS scope=one-after-site-plus-its-projections-not-model\n",saved*90.0/1e6);
        print!("{line}");report.push_str(&line);fs::write(self.out.join(format!("{label}.txt")),report).map_err(|error|error.to_string())
    }
    fn compare(&self,index:usize,sample:&Sample,label:&str,repeat:usize,arm:&str)->Result<()> {
        let case=&self.cases[index];let expected=&self.reference[index];
        if sample.planes.len()!=expected.len() {return Err("plane count differs".into());}
        for (plane,(actual,reference)) in sample.planes.iter().zip(expected).enumerate() {
            let prefix=format!("{label}-{}-{repeat}-{arm}-plane{plane}",case.name);
            fs::write(self.out.join(format!("{prefix}.f32")),actual).map_err(|error|error.to_string())?;
            if let Some(byte)=first_difference(reference,actual) {return Err(format!("BYTE FAIL label={label} case={} arm={arm} repeat={repeat} plane={plane} first_byte={byte} expected={:?} actual={:?} expected_len={} actual_len={}",case.name,reference.get(byte),actual.get(byte),reference.len(),actual.len()));}
        }Ok(())
    }
    fn trial(&self,variant:&sweep::Variant,selected:&[usize],repeats:usize)->Result<f64> {
        let label=&variant.label;let image=&variant.image;
        sweep::label(label)?;check_time(self.deadline)?;
        if repeats<3 || repeats>31 || repeats%2==0 {return Err("repeats must be odd and within 3..31".into());}
        let image=fs::read(image).map_err(|error|error.to_string())?;let started=Instant::now();self.backend.fixture_replace_candidate(&image)?;
        self.backend.fixture_set_geometry(variant.geometry.clone())?;
        println!("IMAGE label={label} sha256={} load_ms={:.6} geometry={:?}",sha256::digest(&image),started.elapsed().as_secs_f64()*1000.0,variant.geometry);
        let mut ratios=Vec::new();let mut report=String::new();
        for &index in selected {
            let mut timings=[Vec::new(),Vec::new()];
            for arm in [false,true] {let sample=self.sample(index,arm)?;self.compare(index,&sample,label,0,if arm {"candidate"} else {"baseline"})?;}
            for repeat in 1..=repeats {for candidate in if repeat%2==1 {[false,true]} else {[true,false]} {
                let arm=if candidate {"candidate"} else {"baseline"};let sample=self.sample(index,candidate)?;self.compare(index,&sample,label,repeat,arm)?;
                if sample.kernel_ns==0 {return Err("GPU event timing returned zero".into());}
                timings[usize::from(candidate)].push(sample.kernel_ns);
                let line=format!("SAMPLE label={label} case={} repeat={repeat} arm={arm} kernel_ns={} wall_ns={} launches={} bytes={} PASS\n",self.cases[index].name,sample.kernel_ns,sample.wall_ns,sample.launches,sample.planes.iter().map(Vec::len).sum::<usize>());
                print!("{line}");report.push_str(&line);
            }}
            let baseline=median(&timings[0])?;let candidate=median(&timings[1])?;let ratio=baseline as f64/candidate as f64;ratios.push(ratio.ln());
            let line=format!("RESULT label={label} case={} token_rows={} matrices=1 macs={} baseline_ns={baseline} candidate_ns={candidate} speedup={ratio:.9} mac_per_second={:.3} baseline_mac_per_second={:.3} repeats={repeats} BYTE_PASS\n",
                self.cases[index].name,self.cases[index].tokens,self.cases[index].macs,self.cases[index].macs as f64*1e9/candidate as f64,self.cases[index].macs as f64*1e9/baseline as f64);
            print!("{line}");report.push_str(&line);
        }
        self.backend.select_arm(false)?;fs::write(self.out.join(format!("{label}.txt")),report).map_err(|error|error.to_string())?;
        Ok((ratios.iter().sum::<f64>()/ratios.len() as f64).exp())
    }
    fn command(&self,line:&str,seen:&mut BTreeSet<String>)->Result<bool> {
        let fields=line.split_whitespace().collect::<Vec<_>>();let Some(command)=fields.first() else {return Ok(true);};
        let names=self.cases.iter().map(|case|case.name.clone()).collect::<Vec<_>>();
        match *command {
            "quit" if fields.len()==1=>return Ok(false),
            "addlat" if fields.len()==1=>self.add_latency()?,
            "shadow" if fields.len()==7=>{
                let number=|text:&str|text.parse::<u64>().map_err(|_|format!("invalid shadow number {text}"));
                sweep::label(fields[1])?;if !seen.insert(fields[1].to_owned()) {return Err(format!("variant label was already used: {}",fields[1]));}
                self.shadow_trial(fields[1],number(fields[2])? as usize,number(fields[3])? as usize,number(fields[4])? as u32,match fields[5] {"normal"=>false,"last"=>true,_=>return Err("shadow policy is normal or last".into())},number(fields[6])? as usize)?;
            },
            "list" if fields.len()==1=>{println!("CASES {}",names.join(","));},
            "run"|"sweep"=>{
                let (variants,case_text,repeat_text)=if *command=="run" && fields.len()>=5 {(vec![sweep::Variant {label:fields[1].to_owned(),image:PathBuf::from(fields[2]),geometry:sweep::geometry(&fields[5..])?}],fields[3],fields[4])}
                    else if *command=="sweep" && fields.len()==4 {(sweep::variants(Path::new(fields[1]))?,fields[2],fields[3])}
                    else {return Err("run LABEL IMAGE CASES REPEATS | sweep MANIFEST CASES REPEATS".into());};
                let selected=chosen(case_text,&names)?;let repeats=repeat_text.parse::<usize>().map_err(|_|"invalid repeats")?;let mut ranked=Vec::new();
                for variant in variants {
                    let label=&variant.label;
                    sweep::label(&label)?;
                    if !seen.insert(label.clone()) {return Err(format!("variant label was already used: {label}"));}
                    if Instant::now()>=self.deadline {println!("SKIP label={label} session budget exhausted");break;}
                    match self.trial(&variant,&selected,repeats) {
                        Ok(ratio)=>ranked.push((label.clone(),ratio)),
                        Err(error)=>{println!("REJECT label={label} {error}");fs::write(self.out.join(format!("{label}.REJECT")),&error).map_err(|error|error.to_string())?;
                            self.backend.synchronize()?;self.backend.select_arm(false)?;if let Some(fatal)=self.backend.fatal_error() {return Err(fatal);}
                            if Instant::now()>=self.deadline {break;}
                        },
                    }
                }
                ranked.sort_by(|left,right|right.1.total_cmp(&left.1));let mut ranking=String::new();
                for (index,(label,ratio)) in ranked.iter().enumerate() {let line=format!("RANK place={} label={label} geometric_speedup={ratio:.9} BYTE_PASS scope=selected-cases-not-model\n",index+1);print!("{line}");ranking.push_str(&line);}
                fs::OpenOptions::new().create(true).append(true).open(self.out.join("ranking.txt")).and_then(|mut file|file.write_all(ranking.as_bytes())).map_err(|error|error.to_string())?;
            },_=>return Err("unknown command; use list, run, sweep, addlat, shadow LABEL DOTS_BYTES FINISH_BYTES BLOCKS normal|last REPEATS, or quit".into()),
        }Ok(true)
    }
}
pub(super) fn serve(checkpoint:&CheckpointAdmission,selections:&[Selection],parts:u32,deadline:Instant,out:&Path,baseline:Option<&Path>,cold_bytes:usize,packed_tokens:u32)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let started=Instant::now();
    with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
        let baseline_identity=if let Some(path)=baseline {
            let image=fs::read(path).map_err(|error|error.to_string())?;backend.fixture_replace_baseline(&image)?;
            format!("baseline_image_sha256={} baseline_external=true",sha256::digest(&image))
        } else {format!("baseline_ptx_sha256={} {} baseline_external=false",sha256::digest(spark_gpu_probe::layer_ptx()?.to_bytes()),spark_gpu_probe::CUBIN_STAMP)};
        backend.configure_vector_pool(64<<20);let cases=load(backend,checkpoint,selections,parts,deadline,packed_tokens)?;
        let lut=backend.upload_vector(&spark_gpu_probe::expert_ops::decode_lut())?;backend.synchronize()?;backend.set_profiling(true);
        let pressure=if cold_bytes==0 {None} else {
            check_time(deadline)?;let words=cold_bytes/4;
            let values=(0..words).map(|index|f32::from_bits(0x3f000000|((index as u32).wrapping_mul(0x9e3779b9)&0x007fffff))).collect::<Vec<_>>();
            let source=backend.upload_vector(&values)?;let output=RefCell::new(backend.vector(words)?);
            let shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::StreamMean,heads:1,dim:words as u32,taps:0,rows:0,cols:0,epsilon:0.0,bound:0.0};
            Some(CachePressure {source,output,shape})
        };
        let mut bench=Bench {backend,cases,reference:Vec::new(),lut,deadline,out,pressure,checkpoint,parts,shadow:RefCell::new(None)};
        for index in 0..bench.cases.len() {let sample=bench.sample(index,false)?;
            for (plane,values) in sample.planes.iter().enumerate() {fs::write(out.join(format!("reference-{}-plane{plane}.f32",bench.cases[index].name)),values).map_err(|error|error.to_string())?;}
            bench.reference.push(sample.planes);
        }
        let identity=format!("{baseline_identity} parts={parts} rank=0 real_weights=true inputs=synthetic-deterministic packed_token_rows={packed_tokens} cases={} cache_pressure_bytes={cold_bytes} startup_seconds={:.6}\n",bench.cases.len(),started.elapsed().as_secs_f64());
        fs::write(out.join("IDENTITY"),&identity).map_err(|error|error.to_string())?;print!("READY {identity}");io::stdout().flush().map_err(|error|error.to_string())?;
        let (sender,receiver)=mpsc::sync_channel(8);std::thread::spawn(move|| {
            for line in io::stdin().lock().lines() {if sender.send(line.map_err(|error|error.to_string())).is_err() {break;}}
        });let mut seen=BTreeSet::new();
        loop {
            let Some(remaining)=deadline.checked_duration_since(Instant::now()) else {println!("SESSION budget reached");break;};
            match receiver.recv_timeout(remaining) {
                Ok(Ok(line))=>{match bench.command(&line,&mut seen) {Ok(true)=>{},Ok(false)=>break,Err(error)=>return Err(error)};io::stdout().flush().map_err(|error|error.to_string())?;},
                Ok(Err(error))=>return Err(error),Err(mpsc::RecvTimeoutError::Disconnected)=>break,Err(mpsc::RecvTimeoutError::Timeout)=>{println!("SESSION idle budget reached");break;},
            }
        }
        backend.synchronize()?;if let Some(error)=backend.fatal_error() {return Err(error);}Ok(())
    })?;
    println!("SESSION COMPLETE seconds={:.6} gpu_context_released=true",started.elapsed().as_secs_f64());fs::write(out.join("COMPLETE"),b"KERNEL BENCH SESSION COMPLETE; inspect each variant result, rejected variants are not PASS\n").map_err(|error|error.to_string())?;Ok(())
}
