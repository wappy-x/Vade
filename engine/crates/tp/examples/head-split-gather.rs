#[cfg(target_os="linux")]
fn run()->Result<(),String> {
    use engine_model::plan::{ConfigPlan,layer::Topology};
    use spark_gpu_probe::driver::with_backend;
    use spark_tp::{resident_weights::{gather_stage,Rows},sparse_heads,tensor_wire,transport::Ring};
    use std::{net::SocketAddr,sync::atomic::{AtomicBool,Ordering},time::{Duration,Instant}};
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=4 {return Err("usage: head-split-gather RANK LISTEN NEXT CONFIG".into());}
    let rank=args[0].parse::<usize>().map_err(|_|"invalid rank")?;
    let listen=args[1].parse::<SocketAddr>().map_err(|_|"invalid listen address")?;
    let next=args[2].parse::<SocketAddr>().map_err(|_|"invalid next address")?;
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(&std::fs::read(&args[3]).map_err(|error|error.to_string())?).map_err(|error|error.to_string())?
        else {return Err("gather fixture requires a described model".into());};
    let topology=Topology {rank:rank as u32,parts:2};
    let layouts=sparse_heads::layouts(&description,topology)?;
    if layouts.is_empty() {return Err("gather fixture has no latent layers".into());}
    let ring=Ring::connect(rank,2,listen,next)?;
    let port=|mut address:SocketAddr|->Result<SocketAddr,String> {address.set_port(address.port().checked_add(1).ok_or("wire port overflow")?);Ok(address)};
    let (incoming,outgoing)=spark_seqpar::control::connect_peers((rank>0).then(||port(listen)).transpose()?,
        (rank==0).then(||port(next)).transpose()?,Instant::now()+Duration::from_secs(300))?;
    let link=incoming.or(outgoing).ok_or("gather fixture wire has no peer")?;
    let wire=tensor_wire::Wire::Write(tensor_wire::WriteLink::new(&link)?);
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|backend| {
        let mailbox=backend.tp_mailbox(4<<20)?;
        let worker=wire.bind(mailbox.take_host().ok_or("gather fixture mailbox is taken")?,rank==0,mailbox.layout())?;
        let (stop,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
        std::thread::scope(|scope| {
            let handle=tensor_wire::spawn(scope,worker,&stop,&attached,&failed)?;
            let result=(|| {
                tensor_wire::wait_attached(&attached,&handle)?;
                let queue=backend.resident_queue()?;backend.enable_resident_launches()?;
                let mut sequence=0;
                let mut local=Vec::new();let mut rows=Vec::new();let mut expected=Vec::new();
                for (layer,layout) in &layouts {
                    let count=layout.local_heads()*layout.value_dim();
                    let payload=|origin:usize| (0..count).map(|at|((at+origin*count+layer)%127) as f32/128.0).collect::<Vec<_>>();
                    local.push(vec![backend.upload_vector(&payload(rank))?]);
                    rows.push(vec![Rows {start:layout.attended_rows().start,count,total:layout.total_heads()*layout.value_dim(),split:true}]);
                    expected.push((0..2).flat_map(payload).map(f32::to_bits).collect::<Vec<_>>());
                }
                backend.synchronize()?;
                for index in 0..layouts.len() {
                    let output=gather_stage(backend,&mailbox,&mut sequence,topology,&local[index],&rows[index])?;
                    let got=backend.read_vector_values(&output[0])?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                    if got!=expected[index] {return Err(format!("gather bytes differ layer={}",layouts[index].0));}
                }
                for context in [1040,4158] {
                    for _ in 0..16 {for index in 0..layouts.len() {let _=gather_stage(backend,&mailbox,&mut sequence,topology,&local[index],&rows[index])?;}backend.synchronize()?;}
                    let mut samples=Vec::new();
                    for _ in 0..128 {
                        backend.synchronize()?;let started=Instant::now();
                        for index in 0..layouts.len() {let _=gather_stage(backend,&mailbox,&mut sequence,topology,&local[index],&rows[index])?;}
                        backend.synchronize()?;mailbox.require_success()?;samples.push(started.elapsed().as_nanos());
                    }
                    samples.sort_unstable();let median=samples[samples.len()/2];
                    let bytes=rows.iter().map(|row|row[0].count*4).sum::<usize>();
                    println!("HEAD_SPLIT_GATHER rank={rank} context={context} synthetic=1 count={} samples={} bytes_per_step={bytes} median_step_ns={median} median_gather_us={:.6} total_step_ms={:.6} min_step_ns={} max_step_ns={} differing_words=0",layouts.len(),samples.len(),median as f64/1000.0/layouts.len() as f64,median as f64/1e6,samples[0],samples[samples.len()-1]);
                    for (layer,layout) in &layouts {println!("HEAD_SPLIT_PAYLOAD layer={layer} local_heads={} value_dim={} wire_bytes={} bf16_logical_bytes={}",layout.local_heads(),layout.value_dim(),layout.local_heads()*layout.value_dim()*4,layout.local_heads()*layout.value_dim()*2);}
                }
                drop(queue);drop(ring);Ok(())
            })();
            if result.is_err() {mailbox.fail();}
            stop.store(true,Ordering::Release);
            let joined=handle.join().map_err(|_|"gather fixture wire panicked".to_owned()).and_then(|result|result);
            result.and(joined)
        })
    })
}
fn main() {
    #[cfg(target_os="linux")]
    if let Err(error)=run() {eprintln!("{error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("head-split-gather requires Linux and CUDA");std::process::exit(1);}
}
