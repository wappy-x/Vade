use std::{path::Path,rc::Rc,sync::{Arc,atomic::{AtomicBool,Ordering}},time::{Duration,Instant}};
use engine_model::plan::{admission::CheckpointAdmission,layer::{Kind,Topology},state::StatePlan};
use spark_gpu_probe::{driver::{Backend,DeviceVector,MhcBoundaryPass,MhcHostLink,ResidentMhcBoundary,Shadow},mhc_host::{HostBoundary,Planes,Pool},mhc_host_service::{serve,Queue},
    mhc_params::{SHADOW_DECODE_BLOCKS,SHADOW_DECODE_STAGE_BYTES},tp_mailbox_host::{ERROR,SEND_LEN,SEND_SEQ}};
use spark_tp::{layer_driver::LayerBoundary,resident_weights::Weights};
type Result<T>=std::result::Result<T,String>;
struct Case<'b,'d> {name:String,matrix:ResidentMhcBoundary<'b,'d>,
    /// The same boundary on the production host lane (chain mode only).
    hosted:Option<ResidentMhcBoundary<'b,'d>>,host:Arc<HostBoundary>,previous:Planes,input:Rc<DeviceVector<'d>>,block:Rc<DeviceVector<'d>>,consumer:Vec<String>}
struct Sample {boundary:u128,consumer:u128,total:u128,planes:Vec<Vec<u32>>}
fn input(words:usize,seed:usize)->Vec<f32> {(0..words).map(|index|((index*13+index/7+seed*19)%61) as f32/32.0-0.9375).collect()}
fn open(root:&Path)->Result<CheckpointAdmission> {
    use engine_model::{plan::ConfigPlan,safetensors::read_bounded,shapes::{CONFIG_BYTES,HEADER_BYTES}};
    let config=read_bounded(&root.join("config.json"),CONFIG_BYTES).map_err(|error|error.to_string())?;
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&config).map_err(|error|error.to_string())? else {return Err("deferred checkpoint required".into());};
    let index=read_bounded(&root.join("model.safetensors.index.json"),HEADER_BYTES).map_err(|error|error.to_string())?;
    description.open_headers(root,&index).map_err(|error|error.to_string())
}
fn load<'b,'d>(backend:&'b Backend<'d>,checkpoint:&CheckpointAdmission,limit:usize,cores:usize,link:Option<&MhcHostLink<'b,'d>>)->Result<(Vec<Case<'b,'d>>,Weights<'b,'d>)> {
    let topology=Topology {rank:0,parts:2};let mut cases=Vec::new();let mut names=Vec::new();
    let state=StatePlan::new(checkpoint.description(),topology,1,isize::MAX as usize).map_err(|error|error.to_string())?;
    for layer in 0..checkpoint.description().layers.len() {
        let boundary=LayerBoundary::new(checkpoint.description(),checkpoint.headers(),&state,layer,1,topology,usize::MAX)?;
        for (index,step) in boundary.program().steps.iter().enumerate().filter(|(_,step)|step.kind==Kind::MhcPrepare) {
            if cases.len()>=limit {break;}
            let operands=boundary.prepare_residual_block_checkpoint(index,checkpoint)?;let mhc=operands.mhc();let norm=operands.norm();
            let normalized=&boundary.program().steps[index+2].outputs[0].name;
            let consumer=boundary.program().steps[index+3..].iter().take_while(|step|step.kind!=Kind::MhcPrepare)
                .filter(|step|step.kind==Kind::Project && step.inputs.len()==1 && &step.inputs[0]==normalized && step.weights.len()==1)
                .map(|step|step.weights[0].name.clone()).collect::<Vec<_>>();
            if consumer.is_empty() {return Err(format!("layer={layer} site={index}: no direct projection consumer"));}
            let source=input(mhc.shape().wide(),cases.len());
            let host=Arc::new(HostBoundary::new(mhc.shape(),norm.shape(),mhc.weights(),mhc.base(),mhc.scale(),norm.weight(),cores)?);
            let mut previous=host.planes();host.run(None,&source,&mut previous);
            names.extend(consumer.iter().cloned());
            println!("LOAD site={} layer={layer} boundary={} consumers={consumer:?}",cases.len(),step.id);
            let hosted=link.map(|link|backend.resident_mhc_boundary_hosted(mhc.shape(),norm.shape(),mhc.weights(),mhc.base(),mhc.scale(),norm.weight(),link)).transpose()?;
            cases.push(Case {name:format!("{layer}:{}",step.id),matrix:operands.upload_boundary(backend)?,hosted,host,previous,
                input:Rc::new(backend.upload_vector(&source)?),block:Rc::new(backend.upload_vector(&input(mhc.shape().hidden as usize,cases.len()+101))?),consumer});
        }
    }
    let weights=Weights::load_local_fixture(checkpoint,backend,topology,&names)?;
    println!("LOAD PASS sites={} projection_names={} device_bytes={} input=synthetic_deterministic weights=checkpoint rank=0 parts=2",cases.len(),names.len(),backend.allocated_bytes());
    Ok((cases,weights))
}
fn bits(backend:&Backend<'_>,vector:&DeviceVector<'_>)->Result<Vec<u32>> {Ok(backend.read_vector_values(vector)?.into_iter().map(f32::to_bits).collect())}
fn sweep<'b,'d>(backend:&'b Backend<'d>,cases:&[Case<'b,'d>],weights:&mut Weights<'b,'d>,arm:usize,blocks:u32,helpers:usize)->Result<Vec<Sample>> {
    let max_words=cases.iter().map(|case|case.host.shape().wide()+case.host.shape().coefficients()).max().ok_or("empty fixture")?;
    let mailbox=backend.tp_mailbox(max_words*4)?;let mut host=mailbox.take_host().ok_or("missing mailbox host")?;host.set_wide(true);host.set_take_pause(128);
    let layout=mailbox.layout();let stop=AtomicBool::new(false);
    let host_cases=cases.iter().map(|case|(case.host.clone(),case.previous.clone())).collect::<Vec<_>>();
    let mut firsts=Vec::new();
    {let queue=backend.resident_queue()?;for case in cases {firsts.push(case.matrix.begin(case.input.clone())?);}drop(queue);}
    backend.synchronize()?;
    std::thread::scope(|scope|->Result<Vec<Sample>> {
        let (ready_send,ready_recv)=std::sync::mpsc::sync_channel(1);
        let stopped=&stop;
        let worker=scope.spawn(move ||->Result<()> {
            let _=spark_gpu_probe::driver::prefer_fast_cores();
            let count=host_cases.iter().map(|(boundary,_)|boundary.shape().coefficients()).max().unwrap();
            let pool=Pool::new(helpers,count);let memory=host.words();ready_send.send(()).map_err(|error|error.to_string())?;
            if arm==0 {return Ok(());}
            let result=(|| {for (index,(boundary,previous)) in host_cases.iter().enumerate() {
                let sequence=index+1;let slot=sequence%layout.slots;let deadline=Instant::now()+Duration::from_secs(10);
                let mut planes=boundary.planes();let mut source=vec![0.0;boundary.shape().hidden as usize];
                while memory[SEND_SEQ+slot].load(Ordering::Acquire)!=sequence as u32 {
                    if stopped.load(Ordering::Acquire) || Instant::now()>=deadline {return Err(format!("host wait timed out at site {index}"));}std::hint::spin_loop();
                }
                if memory[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=source.len()*4 {return Err("host payload length differs".into());}
                pool.prepare(boundary)?;
                let from=layout.send_slot(slot)+layout.frame;
                for (word,value) in source.iter_mut().enumerate() {*value=f32::from_bits(memory[from+word].load(Ordering::Relaxed));}
                if arm==1 {pool.fold_boundary(boundary,Some((&previous.residual,&previous.coefficients)),&source,&mut planes)?;}
                else {pool.run(boundary,Some((&previous.residual,&previous.coefficients)),&source,&mut planes)?;}
                let to=layout.recv_slot(0,slot)+layout.frame;
                if arm==1 {for (word,value) in planes.residual.iter().chain(&planes.coefficients).enumerate() {memory[to+word].store(value.to_bits(),Ordering::Relaxed);}}
                else {for (word,value) in planes.output.iter().enumerate() {memory[to+word].store(value.to_bits(),Ordering::Relaxed);}}
                memory[layout.recv_seq(0,slot)].store(sequence as u32,Ordering::Release);
            }Ok(())})();
            if result.is_err() {memory[ERROR].store(1,Ordering::Release);}result
        });
        ready_recv.recv().map_err(|error|error.to_string())?;
        let queued=(|| {
            let queue=backend.resident_queue()?;let mut completed=Vec::new();
            for (index,(case,first)) in cases.iter().zip(firsts).enumerate() {
                let shape=case.host.shape();let wide=shape.wide();let count=shape.coefficients();let hidden=shape.hidden as usize;
                let mut residual=backend.vector(wide)?;let mut raw=backend.vector(count)?;let mut output=backend.vector(hidden)?;
                let mut spans=[None,None,None];let (mut target,mut offset)=(0,0);
                for (span,budget) in spans.iter_mut().zip(SHADOW_DECODE_STAGE_BYTES) {
                    if budget==0 {continue;}
                    while let Some(name)=case.consumer.get(target) {match weights.shadow_range(backend,name,offset,budget)? {
                        Some((range,total))=>{offset+=range.bytes() as usize;if offset>=total {target+=1;offset=0;}*span=Some(range);break;},
                        None=>{target+=1;offset=0;},
                    }}
                }
                let shadow=Shadow {rms:spans[0],dots:spans[1],finish:spans[2],blocks,evict_last:false};
                let start=backend.gpu_mark()?;
                let pass:Option<MhcBoundaryPass<'b,'d>>=if arm==0 {Some(case.matrix.begin_after_shadowed(first,case.block.clone(),Rc::new(residual),Some(&shadow))?)}
                else {
                    backend.tp_mailbox_put(&mailbox,&case.block,1,hidden,hidden,0,(index+1) as u64)?;
                    backend.chain_shadow_prefetch(&Shadow {finish:if arm==1 {None} else {shadow.finish},..shadow})?;
                    if arm==1 {
                        backend.tp_mailbox_take_pair_from(&mailbox,[(&mut residual,wide,wide,0),(&mut raw,count,count,0)],1,0,(index+1) as u64,10_000_000_000)?;
                        Some(case.matrix.finish_host_planes(residual,raw,Some(&shadow))?)
                    } else {backend.tp_mailbox_take(&mailbox,&mut output,1,hidden,hidden,0,(index+1) as u64,10_000_000_000)?;None}
                };
                let normalized=pass.as_ref().map_or_else(||Rc::new(output),MhcBoundaryPass::normalized);
                let middle=backend.gpu_mark()?;
                let mut projected=Vec::new();
                for name in &case.consumer {projected.push(weights.apply(name,backend,&normalized,false)?);}
                let end=backend.gpu_mark()?;
                completed.push((pass,normalized,projected,[start,middle,end]));
            }
            drop(queue);backend.synchronize()?;mailbox.require_success()?;
            let mut samples=Vec::new();
            for (pass,normalized,projected,marks) in completed {
                let mut planes=vec![bits(backend,&normalized)?];for vector in projected {planes.push(bits(backend,&vector)?);}
                if let Some(pass)=pass {planes.push(bits(backend,pass.residual())?);planes.push(bits(backend,pass.coefficients())?);}
                samples.push(Sample {boundary:marks[0].ns_until(&marks[1])?,consumer:marks[1].ns_until(&marks[2])?,total:marks[0].ns_until(&marks[2])?,planes});
            }Ok(samples)
        })();
        if queued.is_err() {stop.store(true,Ordering::Release);mailbox.fail();}
        worker.join().map_err(|_|"host worker panicked")??;queued
    })
}
/// Chain mode: the 90 boundaries as ONE chain, as a decode step runs them (each boundary takes the last
/// boundary's residual and coefficients), with every dependent projection between two boundaries.
/// `hosted` = the production host lane (job queue, service thread, put, prefetch launches, take);
/// else the kernels with their CHAIN-SHADOW tails. Every launch is queued ahead, as in a step.
fn chain_sweep<'b,'d>(backend:&'b Backend<'d>,cases:&[Case<'b,'d>],weights:&mut Weights<'b,'d>,hosted:bool,blocks:u32,recorded:bool)->Result<Vec<Sample>> {
    let queue=backend.resident_queue()?;
    // `recorded`: the launches go through the resident recorder and run as graph pieces, as a persistent decode step does.
    if recorded {backend.enable_resident_launches()?;}let mut completed=Vec::new();let mut last:Option<MhcBoundaryPass<'b,'d>>=None;
    for case in cases {
        let matrix=if hosted {case.hosted.as_ref().ok_or("chain mode has no hosted boundary")?} else {&case.matrix};let wide=case.host.shape().wide();
        let mut spans=[None,None,None];let (mut target,mut offset)=(0,0);
        for (span,budget) in spans.iter_mut().zip(SHADOW_DECODE_STAGE_BYTES) {
            if budget==0 {continue;}
            while let Some(name)=case.consumer.get(target) {match weights.shadow_range(backend,name,offset,budget)? {
                Some((range,total))=>{offset+=range.bytes() as usize;if offset>=total {target+=1;offset=0;}*span=Some(range);break;},
                None=>{target+=1;offset=0;},
            }}
        }
        let shadow=Shadow {rms:spans[0],dots:spans[1],finish:spans[2],blocks,evict_last:false};
        let start=backend.gpu_mark()?;
        let pass=match last.take() {None=>matrix.begin_shadowed(case.input.clone(),Some(&shadow))?,
            Some(previous)=>matrix.begin_after_shadowed(previous,case.block.clone(),Rc::new(backend.vector(wide)?),Some(&shadow))?};
        if hosted && pass.hosted_take().is_none() {return Err("a chain boundary did not take the host lane".into());}
        let normalized=pass.normalized();let middle=backend.gpu_mark()?;
        let mut projected=Vec::new();
        for name in &case.consumer {projected.push(weights.apply(name,backend,&normalized,false)?);}
        let end=backend.gpu_mark()?;completed.push((normalized,projected,[start,middle,end]));last=Some(pass);
    }
    drop(queue);backend.synchronize()?;drop(last);
    let mut samples=Vec::new();
    for (normalized,projected,marks) in completed {
        let mut planes=vec![bits(backend,&normalized)?];for vector in projected {planes.push(bits(backend,&vector)?);}
        samples.push(Sample {boundary:marks[0].ns_until(&marks[1])?,consumer:marks[1].ns_until(&marks[2])?,total:marks[0].ns_until(&marks[2])?,planes});
    }Ok(samples)
}
fn chain_run(backend:&Backend<'_>,checkpoint:&CheckpointAdmission,limit:usize,repeats:usize,blocks:u32,helpers:usize,recorded:bool)->Result<()> {
    let mailbox=backend.tp_mailbox(1<<20)?;let mut host=mailbox.take_host().ok_or("missing mailbox host")?;host.set_wide(true);host.set_take_pause(128);
    let queue=Queue::new();let link=MhcHostLink::new(&mailbox,queue.clone(),helpers+1);
    let (cases,mut weights)=load(backend,checkpoint,limit,helpers+1,Some(&link))?;
    std::thread::scope(|scope|->Result<()> {
        let service=scope.spawn(||serve(host,&queue,helpers));
        let result=(|| {
            let mut totals=[Vec::new(),Vec::new()];let mut parts=[[0u128;2];2];
            for repeat in 0..=repeats {for arm in if repeat%2==0 {[0,1]} else {[1,0]} {
                let before=backend.stats();let samples=chain_sweep(backend,&cases,&mut weights,arm==1,blocks,recorded)?;let stats=queue.take_stats();
                let kinds=backend.stats().since(before).recorded_kinds;
                if arm==1 {let reference=chain_sweep(backend,&cases,&mut weights,false,blocks,false)?;
                    for (site,(sample,want)) in samples.iter().zip(&reference).enumerate() {for (plane,(got,want)) in sample.planes.iter().zip(&want.planes).enumerate() {
                        if got!=want {let word=got.iter().zip(want).position(|(got,want)|got!=want);return Err(format!("BYTE FAIL chain site={site} plane={plane} word={word:?}"));}}}
                    queue.take_stats();}
                let sum=samples.iter().map(|sample|sample.total).sum::<u128>();
                let (boundary,consumer)=(samples.iter().map(|sample|sample.boundary).sum::<u128>(),samples.iter().map(|sample|sample.consumer).sum::<u128>());
                println!("CHAIN SWEEP repeat={repeat} arm={} sites={} total_ns={sum} boundary_us_per_site={:.1} consumer_us_per_site={:.1}{}",if arm==1 {"host_lane"} else {"kernels"},samples.len(),
                    boundary as f64/1e3/samples.len() as f64,consumer as f64/1e3/samples.len() as f64,if arm==1 {" BYTE_PASS"} else {""});
                if arm==1 && stats[0]>0 {let per=|i:usize|stats[i] as f64/1e3/stats[0] as f64;
                    println!("CHAIN HOST PARTS us jobs={} wait_put={:.1} copy_in={:.1} residual={:.1} rms_chain={:.1} folds={:.1} finish={:.1} answer={:.1}",stats[0],per(1),per(2),per(3),per(4),per(5),per(6),per(7));}
                if recorded {use spark_gpu_probe::resident_launch::Kind;
                    println!("CHAIN RECORDED arm={} put_plain={} shadow_prefetch={} take={} mhc_rms={} mhc_dots={} mhc_finish={} all_kinds={}",if arm==1 {"host_lane"} else {"kernels"},kinds[Kind::PutPlain as usize],kinds[Kind::ShadowPrefetch as usize],
                        kinds[Kind::Take as usize],kinds[Kind::MhcBoundaryRms as usize],kinds[Kind::MhcBoundaryDots as usize],kinds[Kind::MhcBoundaryFinish as usize],kinds.iter().sum::<u128>());}
                if repeat>0 {totals[arm].push(sum);parts[arm][0]+=boundary;parts[arm][1]+=consumer;}
            }}
            for values in &mut totals {values.sort_unstable();}
            let median=[totals[0][totals[0].len()/2],totals[1][totals[1].len()/2]];let saved=(median[0] as f64-median[1] as f64)/cases.len() as f64;let n=(repeats*cases.len()) as f64;
            println!("CHAIN RESULT recorded={recorded} sites={} blocks={blocks} helpers={helpers} kernels_ns={} host_lane_ns={} saved_ns_per_boundary={saved:.1} verdict={} kernels boundary/consumer us={:.1}/{:.1} host_lane boundary/consumer us={:.1}/{:.1} scope=single_rank_chain_of_boundaries_plus_dependent_projections_not_model",
                cases.len(),median[0],median[1],if saved>=30_000.0 {"PASS"} else {"KILL"},parts[0][0] as f64/1e3/n,parts[0][1] as f64/1e3/n,parts[1][0] as f64/1e3/n,parts[1][1] as f64/1e3/n);
            if mailbox.failed() || queue.failed() {return Err("host lane failed".into());}Ok(())
        })();
        queue.stop();let served=service.join().map_err(|_|"service thread panicked".to_string()).and_then(|result|result);result.and(served)
    })
}
pub fn run()->Result<()> {
    let args=std::env::args().collect::<Vec<_>>();let root=args.get(1).ok_or("usage: conductor-net MODEL [sites] [repeats] [blocks] [helpers]")?;
    let number=|index:usize,default:usize|->Result<usize> {args.get(index).map_or(Ok(default),|value|value.parse().map_err(|_|"invalid integer".into()))};
    let limit=number(2,usize::MAX)?;let repeats=number(3,5)?;let blocks=number(4,SHADOW_DECODE_BLOCKS as usize)? as u32;let helpers=number(5,2)?;
    if limit==0 || repeats==0 || repeats>15 || helpers>3 {return Err("invalid fixture bounds".into());}
    let checkpoint=open(Path::new(root))?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
        if let Some(mode)=args.get(6).filter(|mode|*mode=="chain" || *mode=="chain-recorded") {return chain_run(backend,&checkpoint,limit,repeats,blocks,helpers,mode=="chain-recorded");}
        let (cases,mut weights)=load(backend,&checkpoint,limit,helpers+1,None)?;let mut reference:Option<Vec<Sample>>=None;let mut totals=[Vec::new(),Vec::new(),Vec::new()];
        for repeat in 0..=repeats {for arm in if repeat%2==0 {[0,1,2]} else {[2,1,0]} {
            let samples=sweep(backend,&cases,&mut weights,arm,blocks,helpers)?;
            if let Some(reference)=&reference {for (site,(sample,want)) in samples.iter().zip(reference).enumerate() {
                for (plane,(got,want)) in sample.planes.iter().zip(&want.planes).enumerate() {
                    if got!=want {let word=got.iter().zip(want).position(|(got,want)|got!=want);return Err(format!("BYTE FAIL arm={arm} site={site} plane={plane} word={word:?}"));}
                }
            }}
            let sum=samples.iter().map(|sample|sample.total).sum::<u128>();
            for (site,sample) in samples.iter().enumerate() {println!("SAMPLE repeat={repeat} arm={arm} site={site} name={} boundary_ns={} consumer_ns={} total_ns={} BYTE_PASS",cases[site].name,sample.boundary,sample.consumer,sample.total);}
            println!("SWEEP repeat={repeat} arm={arm} sites={} total_ns={sum} mean_ns={:.1}",samples.len(),sum as f64/samples.len() as f64);
            if repeat>0 {totals[arm].push(sum);}
            if reference.is_none() {reference=Some(samples);}
        }}
        for values in &mut totals {values.sort_unstable();}
        let median=totals.map(|values|values[values.len()/2]);
        for arm in [1,2] {let saved=(median[0] as f64-median[arm] as f64)/cases.len() as f64;
            println!("RESULT arm={arm} sites={} baseline_ns={} candidate_ns={} saved_ns_per_boundary={saved:.1} verdict={} scope=single_rank_boundary_plus_dependent_projections_not_model",cases.len(),median[0],median[arm],if saved>=30_000.0 {"PASS"} else {"KILL"});}
        Ok(())
    })
}
