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
/// MHC-BATCH premise, GPU form: the chain of boundaries with `rows` rows a boundary (the split kernels the prompt
/// path uses for row groups of 8 or fewer), no prefetch tails, then the first consumer projections on row 0.
fn rows_sweep<'b,'d>(backend:&'b Backend<'d>,cases:&[Case<'b,'d>],inputs:&[(Rc<DeviceVector<'d>>,Rc<DeviceVector<'d>>)],weights:&mut Weights<'b,'d>,rows:usize)->Result<Vec<Sample>> {
    let queue=backend.resident_queue()?;let mut completed=Vec::new();let mut last:Option<MhcBoundaryPass<'b,'d>>=None;
    for (case,(source,block)) in cases.iter().zip(inputs) {
        let (wide,hidden)=(case.host.shape().wide(),case.host.shape().hidden as usize);
        let start=backend.gpu_mark()?;
        let pass=match last.take() {None=>case.matrix.begin_rows(source.clone(),rows)?,
            Some(previous)=>case.matrix.begin_after_rows(previous,block.clone(),Rc::new(backend.vector(rows*wide)?),rows)?};
        let normalized=pass.normalized();let middle=backend.gpu_mark()?;
        let first=normalized.resident_range(0..hidden)?;let mut projected=Vec::new();
        for name in &case.consumer {projected.push(weights.apply(name,backend,&first,false)?);}
        let end=backend.gpu_mark()?;completed.push((normalized,projected,[start,middle,end]));last=Some(pass);
    }
    drop(queue);backend.synchronize()?;drop(last);
    let mut samples=Vec::new();
    for (normalized,_projected,marks) in completed {
        samples.push(Sample {boundary:marks[0].ns_until(&marks[1])?,consumer:marks[1].ns_until(&marks[2])?,total:marks[0].ns_until(&marks[2])?,planes:vec![bits(backend,&normalized)?]});
    }Ok(samples)
}
/// MHC-BATCH premise, CPU form: `rows` chains of the 90 boundaries, each row a whole boundary on one core
/// (HostBoundary::run, the production one-core twin), rows dealt to `cores` pinned workers (fast cores 0..cores of the
/// process mask; the timing thread takes fast core 5). Wall us a site of the work phase. After each site every worker
/// reads the next site's weights (untimed), as the production lane touches weights ahead while the GPU projects.
/// `streaming`: the GPU reads weights (prefetch launches) during the sweep, as it does in a decode step.
#[allow(clippy::too_many_arguments)]
fn host_rows<'b,'d>(backend:&'b Backend<'d>,cases:&[Case<'b,'d>],weights:&Weights<'b,'d>,cores:usize,rows:usize,streaming:bool,blocks:u32,reference:&[Vec<u32>],gpu_rows:usize)->Result<(f64,f64,f64)> {
    use std::sync::atomic::AtomicU32;
    let sites=cases.len();let sources:Vec<Vec<f32>>=cases.iter().enumerate().map(|(index,case)|input(case.host.shape().wide(),index)).collect();
    let blocks_in:Vec<Vec<f32>>=cases.iter().enumerate().map(|(index,case)|input(case.host.shape().hidden as usize,index+101)).collect();
    let hosts:Vec<Arc<HostBoundary>>=cases.iter().map(|case|case.host.clone()).collect();
    let (go,done,stop,equal,start)=(AtomicU32::new(0),AtomicU32::new(0),AtomicBool::new(false),AtomicBool::new(true),AtomicBool::new(false));
    // Diagnostic: the CPU each worker runs on and its busy ns in the work phases.
    let report:Vec<(AtomicU32,std::sync::atomic::AtomicU64)>=(0..cores).map(|_|(AtomicU32::new(u32::MAX),std::sync::atomic::AtomicU64::new(0))).collect();
    let cpu=||std::fs::read_to_string("/proc/thread-self/stat").ok().and_then(|stat|stat.rsplit(')').next().and_then(|rest|rest.split_whitespace().nth(36).and_then(|value|value.parse::<u32>().ok()))).unwrap_or(u32::MAX);
    let result=std::thread::scope(|scope|->Result<(f64,f64,f64)> {
        for core in 0..cores {
            let (go,done,stop,equal,sources,blocks_in,hosts,report)=(&go,&done,&stop,&equal,&sources,&blocks_in,&hosts,&report);
            scope.spawn(move || {
                let _=spark_gpu_probe::driver::prefer_fast_core(core);report[core].0.store(cpu(),Ordering::Release);
                let mine=(0..rows).filter(|row|row%cores==core).count();
                let mut planes:Vec<(Planes,Planes)>=(0..mine).map(|_|(hosts[0].planes(),hosts[0].planes())).collect();
                let mut seen=0u32;
                loop {
                    let now=go.load(Ordering::Acquire);
                    if now==seen {if stop.load(Ordering::Acquire) {return;}std::hint::spin_loop();continue;}
                    seen=now;let site=((now as usize-1)/2)%sites;let host=&hosts[site];
                    if now%2==1 {
                        let busy=Instant::now();
                        for (previous,current) in planes.iter_mut() {
                            if site==0 {host.run(None,&sources[0],current);} else {host.run(Some((&previous.residual,&previous.coefficients)),&blocks_in[site],current);}
                            // The chain's last output depends on every boundary before it: one compare a sweep.
                            if site+1==sites && current.output.iter().map(|value|value.to_bits()).ne(reference[site].iter().copied()) {equal.store(false,Ordering::Release);}
                            std::mem::swap(previous,current);
                        }
                        report[core].1.fetch_add(busy.elapsed().as_nanos() as u64,Ordering::Relaxed);
                    } else if mine>0 {let next=&hosts[(site+1)%sites];for group in 0..next.groups() {std::hint::black_box(next.touch(group));}}
                    done.fetch_add(1,Ordering::Release);
                }
            });
        }
        let timer={let (go,done,start)=(&go,&done,&start);scope.spawn(move || {
            let _=spark_gpu_probe::driver::prefer_fast_core(5);
            while !start.load(Ordering::Acquire) {std::hint::spin_loop();}
            let mut walls=Vec::new();let mut sequence=0u32;
            let phase=|sequence:&mut u32| {*sequence+=1;done.store(0,Ordering::Release);go.store(*sequence,Ordering::Release);while done.load(Ordering::Acquire)!=cores as u32 {std::hint::spin_loop();}};
            for sweep in 0..6 {
                let mut work=Duration::ZERO;
                for _ in 0..sites {let started=Instant::now();phase(&mut sequence);work+=started.elapsed();phase(&mut sequence);}
                if sweep>0 {walls.push(work.as_secs_f64()*1e6/sites as f64);}
            }
            walls.sort_by(f64::total_cmp);(walls[walls.len()/2],walls[0])
        })};
        let queue=if streaming {
            // About a third of a second of weight reading queued ahead of the timed sweeps.
            let queue=backend.resident_queue()?;let mut launched=0;
            'fill: loop {for case in cases {for name in &case.consumer {
                if let Some((range,_))=weights.shadow_range(backend,name,0,7<<20)? {backend.chain_shadow_prefetch(&Shadow {rms:None,dots:Some(range),finish:None,blocks,evict_last:false})?;launched+=1;if launched>=9000 {break 'fill;}}
            }}}
            Some(queue)
        } else {None};
        // The mixed form: chains of `gpu_rows` rows on the split kernels run on the GPU while the CPU rows run.
        let mut chains=Vec::new();
        let mixed=if gpu_rows>0 {
            let inputs=cases.iter().enumerate().map(|(index,case)| {
                let (wide,hidden)=(case.host.shape().wide(),case.host.shape().hidden as usize);
                Ok((Rc::new(backend.upload_vector(&input(wide,index).repeat(gpu_rows))?),Rc::new(backend.upload_vector(&input(hidden,index+101).repeat(gpu_rows))?)))
            }).collect::<Result<Vec<_>>>()?;
            backend.synchronize()?;let queue=backend.resident_queue()?;
            for _ in 0..10 {
                let begin=backend.gpu_mark()?;let mut last:Option<MhcBoundaryPass<'b,'d>>=None;
                for (case,(source,block)) in cases.iter().zip(&inputs) {
                    let wide=case.host.shape().wide();
                    last=Some(match last.take() {None=>case.matrix.begin_rows(source.clone(),gpu_rows)?,Some(previous)=>case.matrix.begin_after_rows(previous,block.clone(),Rc::new(backend.vector(gpu_rows*wide)?),gpu_rows)?});
                }
                chains.push((begin,backend.gpu_mark()?,last));
            }
            Some(queue)
        } else {None};
        start.store(true,Ordering::Release);
        let walls=timer.join().map_err(|_|"timer thread panicked".to_string());
        stop.store(true,Ordering::Release);drop(queue);drop(mixed);backend.synchronize()?;
        let mut gpu=chains.iter().map(|(begin,end,_)|begin.ns_until(end).map(|ns|ns as f64/1e3/sites as f64)).collect::<Result<Vec<_>>>()?;gpu.sort_by(f64::total_cmp);
        walls.map(|(median,least)|(median,least,gpu.get(gpu.len()/4).copied().unwrap_or(0.0)))
    })?;
    if !equal.load(Ordering::Acquire) {return Err(format!("BYTE FAIL host rows cores={cores} rows={rows}: a row's output differs from its solo run"));}
    println!("ROWS CPU DETAIL cores={cores} rows={rows} streaming={streaming} worker cpu:busy_us_a_site {}",report.iter().map(|(cpu,busy)|format!("{}:{:.1}",cpu.load(Ordering::Acquire),busy.load(Ordering::Relaxed) as f64/1e3/(6*sites) as f64)).collect::<Vec<_>>().join(" "));
    Ok(result)
}
fn rows_run(backend:&Backend<'_>,checkpoint:&CheckpointAdmission,limit:usize,repeats:usize,blocks:u32)->Result<()> {
    // Hosts are built for ONE core (one group of 24 coefficient rows): a row is a whole boundary on its core.
    let (cases,mut weights)=load(backend,checkpoint,limit,1,None)?;let sites=cases.len() as f64;
    let mut solo:Vec<Vec<u32>>=Vec::new();
    for rows in [1usize,2,4,8] {
        let inputs=cases.iter().enumerate().map(|(index,case)| {
            let (wide,hidden)=(case.host.shape().wide(),case.host.shape().hidden as usize);
            Ok((Rc::new(backend.upload_vector(&input(wide,index).repeat(rows))?),Rc::new(backend.upload_vector(&input(hidden,index+101).repeat(rows))?)))
        }).collect::<Result<Vec<_>>>()?;
        let mut times=Vec::new();
        for repeat in 0..=repeats {
            let samples=rows_sweep(backend,&cases,&inputs,&mut weights,rows)?;
            if solo.is_empty() {solo=samples.iter().map(|sample|sample.planes[0].clone()).collect();}
            for (site,sample) in samples.iter().enumerate() {for (row,got) in sample.planes[0].chunks_exact(solo[site].len()).enumerate() {
                if got!=solo[site] {return Err(format!("BYTE FAIL gpu rows={rows} site={site} row={row}: the row differs from its solo run"));}}}
            if repeat>0 {times.push((samples.iter().map(|sample|sample.boundary).sum::<u128>(),samples.iter().map(|sample|sample.consumer).sum::<u128>()));}
        }
        times.sort_unstable();let (boundary,consumer)=times[times.len()/2];
        println!("ROWS GPU rows={rows} sites={} boundary_us_per_site={:.1} consumer_row0_us_per_site={:.1} every_row_equals_its_solo_run=true prefetch_tails=none",cases.len(),boundary as f64/1e3/sites,consumer as f64/1e3/sites);
    }
    // The one-row CPU outputs must be the kernels' (the twin): the host reference of each site in the chain.
    let mut reference=Vec::new();{let mut previous=cases[0].host.planes();let mut current=cases[0].host.planes();
        for (site,case) in cases.iter().enumerate() {
            if site==0 {case.host.run(None,&input(case.host.shape().wide(),0),&mut current);} else {case.host.run(Some((&previous.residual,&previous.coefficients)),&input(case.host.shape().hidden as usize,site+101),&mut current);}
            reference.push(current.output.iter().map(|value|value.to_bits()).collect::<Vec<u32>>());std::mem::swap(&mut previous,&mut current);
        }}
    if reference.iter().zip(&solo).any(|(host,kernel)|host!=kernel) {return Err("BYTE FAIL: the one-core host chain differs from the kernel chain".into());}
    println!("ROWS TWIN PASS sites={} one_core_host_chain_equals_kernel_chain=true",cases.len());
    for (streaming,cores,rows,gpu_rows) in [(false,3usize,1usize,0usize),(false,3,2,0),(false,3,3,0),(false,4,4,0),(false,5,5,0),(false,5,8,0),(false,3,2,2),(false,3,3,1),(false,4,4,0),(false,5,4,4),(false,5,5,3),(false,2,2,2)] {{{
        let (median,least,gpu)=host_rows(backend,&cases,&weights,cores,rows,streaming,blocks,&reference,gpu_rows)?;
        println!("ROWS MIXED cpu_rows={rows} on_cores={cores} gpu_rows={gpu_rows} cpu_wall_us_per_site median={median:.1} min={least:.1} gpu_boundary_us_per_site_while_cpu_runs={gpu:.1} total_rows={} every_cpu_row_equals_its_solo_run=true hand_off=not_included",rows+gpu_rows);
    }}}
    println!("ROWS COMPLETE");Ok(())
}
pub fn run()->Result<()> {
    let args=std::env::args().collect::<Vec<_>>();let root=args.get(1).ok_or("usage: conductor-net MODEL [sites] [repeats] [blocks] [helpers]")?;
    let number=|index:usize,default:usize|->Result<usize> {args.get(index).map_or(Ok(default),|value|value.parse().map_err(|_|"invalid integer".into()))};
    let limit=number(2,usize::MAX)?;let repeats=number(3,5)?;let blocks=number(4,SHADOW_DECODE_BLOCKS as usize)? as u32;let helpers=number(5,2)?;
    if limit==0 || repeats==0 || repeats>15 || helpers>3 {return Err("invalid fixture bounds".into());}
    let checkpoint=open(Path::new(root))?;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,usize::MAX,|backend| {
        if let Some(mode)=args.get(6).filter(|mode|*mode=="chain" || *mode=="chain-recorded") {return chain_run(backend,&checkpoint,limit,repeats,blocks,helpers,mode=="chain-recorded");}
        if args.get(6).is_some_and(|mode|mode=="rows") {return rows_run(backend,&checkpoint,limit,repeats,blocks);}
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
