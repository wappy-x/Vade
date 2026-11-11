//! Bounded text service on the shared HTTP scheduler and complete-row transport.
use engine_model::plan::{admission::CheckpointAdmission,layer::Topology};
use spark_serve::{Engine,Tokenizer};
use spark_tp::{Result,row_split::{Factory,Split},bounded_decoder::Device,transport::Ring};
use std::{net::{SocketAddr,TcpListener},path::{Path,PathBuf},sync::atomic::AtomicBool,time::Duration};
pub fn run(args:&[String],checkpoint:CheckpointAdmission,root:&Path,output_setting:Option<String>,memory_setting:Option<String>)->Result<()> {
    let diagnostic=super::settings::diagnostic_text(args.get(9..).unwrap_or_default())?;
    if !diagnostic {checkpoint.description().require_resident_service().map_err(|e|format!("text service refused: {e}; source precision and zero-sign policy require owner approval"))?;}
    if args.len()<9 {return Err("text service needs RANK WORLD LISTEN NEXT PTX MODEL CONTEXT BIND REQUESTS".into());}
    let rank=args[0].parse::<usize>().map_err(|e|e.to_string())?;let world=args[1].parse::<usize>().map_err(|e|e.to_string())?;
    let topology=Topology {rank:rank.try_into().map_err(|_|"rank exceeds topology")?,parts:world.try_into().map_err(|_|"world exceeds topology")?};topology.validate().map_err(|e|e.to_string())?;
    let listen=args[2].parse::<SocketAddr>().map_err(|e|e.to_string())?;let next=args[3].parse::<SocketAddr>().map_err(|e|e.to_string())?;
    let context=args[6].parse::<usize>().map_err(|e|e.to_string())?;let bind=args[7].parse::<SocketAddr>().map_err(|e|e.to_string())?;
    let requests=args[8].parse::<usize>().map_err(|e|e.to_string())?;
    let mut key=None;let mut mtp=None;let mut trace=false;let mut options=args[9..].iter();
    while let Some(option)=options.next() {match option.as_str() {
        "--diagnostic-unqualified-text"=>{},
        "--diagnostic-serving-trace" if !trace=>trace=true,
        "--api-key-file" if key.is_none()=>key=Some(PathBuf::from(options.next().ok_or("--api-key-file needs a path")?)),
        "--mtp" if mtp.is_none()=>{let value=options.next().ok_or("--mtp needs a depth")?;mtp=Some(value.parse::<usize>().ok().filter(|depth|*depth<=3).ok_or("text service --mtp takes a depth of 0 to 3")?);},
        _=>return Err(format!("unsupported text service option: {option}")),
    }}
    super::settings::diagnostic_text_bind(bind)?;
    let auth=spark_serve::auth::ApiKey::load(key.as_deref(),false)?;
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(root)?;
    let bound=spark_serve::text_checkpoint::CheckpointTextEdges::new(&checkpoint)?;
    let eos=tokenizer.checkpoint_eos_token_ids()?.ok_or("text service requires checkpoint EOS metadata")?;
    if eos.len()!=bound.edges().eos_tokens.len() || bound.edges().eos_tokens.iter().any(|id|!eos.contains(id) || !tokenizer.has_token(*id)) {return Err("service and tokenizer EOS sets differ".into());}
    let ptx=spark_gpu_probe::layer_ptx()?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let mut budget=free.saturating_sub((free/10).max(2<<30));
    if let Some(value)=setting("SEQPAR_MEMORY_BYTES",memory_setting)? {budget=budget.min(value);}
    if budget==0 {return Err("text service has no device memory budget".into());}
    let identity=spark_serve::contract::Identity::load(root)?;
    let _cpu_pin=spark_tp::runtime::defer_cpu_pin();
    let mut ring=Ring::connect(rank,world,listen,next)?;
    let wire_topology=ring.topology();let mut records=vec![0;world*80];let local=&mut records[rank*80..(rank+1)*80];
    local[..64].copy_from_slice(identity.fingerprint().as_bytes());local[64..72].copy_from_slice(&(context as u64).to_le_bytes());local[72..].copy_from_slice(&(budget as u64).to_le_bytes());
    ring.all_gather(&mut records,80)?;
    for record in records.as_chunks::<80>().0 {
        if record[..72]!=records[..72] {return Err("text service rank metadata or context differs".into());}
        budget=budget.min(u64::from_le_bytes(record[72..].try_into().unwrap()).try_into().map_err(|_|"rank budget exceeds address space")?);
    }
    spark_gpu_probe::driver::with_backend(0,ptx,budget,|backend| {
        backend.require_resident_stream()?;
        let mut data_listen=listen;data_listen.set_port(listen.port().checked_add(2).ok_or("text data listen port exceeds range")?);
        let mut data_next=next;data_next.set_port(next.port().checked_add(2).ok_or("text data next port exceeds range")?);
        let data=Ring::connect(rank,world,data_listen,data_next)?;
        let mailbox=backend.tp_mailbox(4<<20)?;
        // The host mHC lane takes the rows of a batch step too (one row a core).
        let conductor=spark_tp::host_conductor::Conductor::with_rows(backend,&checkpoint,3,spark_tp::bounded_decoder::BATCH_ROWS)?;
        let mut placement=spark_tp::thread_affinity::Placement::new(&mut ring,conductor.as_ref().map_or(0,|conductor|conductor.cores()))?;
        let physical=Topology {rank:topology.rank,parts:mailbox.layout().world.try_into().map_err(|_|"mailbox topology exceeds ABI")?};let compatible=topology.eq(&physical);
        if !compatible {return Err("diagnostic text service requires exactly 2 Sparks".into());}
        let port=|mut address:SocketAddr|->Result<SocketAddr> {address.set_port(address.port().checked_add(1).ok_or("text wire port exceeds range")?);Ok(address)};
        use spark_gpu_probe::tp_route::{Links,Side};
        let links=Links::new(world,false)?;
        let left=links.neighbor(rank,Side::Left).map(|_|port(data_listen)).transpose()?;
        let right=links.neighbor(rank,Side::Right).map(|_|port(data_next)).transpose()?;let lead=left.is_none();
        let (incoming,outgoing)=spark_seqpar::control::connect_peers(left,right,std::time::Instant::now()+Duration::from_secs(300))?;
        let link=incoming.or(outgoing).ok_or("text resident wire has no peer")?;
        let wire=spark_tp::tensor_wire::Wire::Write(spark_tp::tensor_wire::WriteLink::new(&link)?);
        let worker=wire.bind(mailbox.take_host().ok_or("text mailbox endpoint is taken")?,lead,mailbox.layout())?;
        use std::sync::atomic::Ordering;
        let (stop_wire,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
        std::thread::scope(|scope| {
            let workers=placement.workers();
            let host_worker=conductor.as_ref().map(|conductor|conductor.spawn(scope)).transpose()?;
            let handle=spark_tp::tensor_wire::spawn_on(scope,worker,&stop_wire,&attached,&failed,placement.wire_mask())?;
            drop(workers);
            let result=(|| {
                spark_tp::tensor_wire::wait_attached(&attached,&handle)?;
        let compute=Device::with_mailbox(backend,data,false,&mailbox)?.with_serving_trace(trace);
        let compute=match &conductor {Some(conductor)=>compute.with_mhc_host(conductor.link())?,None=>compute};
        let split=Split::new(compute,ring)?;let engine=Factory::new(&checkpoint,split,context,eos.to_vec())?;
        // The predictor layer is loaded only for --mtp 1..3, before admission counts the resident bytes.
        let engine=match mtp {Some(depth) if depth>0=>engine.with_predictor(depth)?,_=>engine};
        // Load readers keep the initial mask; decode takes its mask once the weights are resident.
        placement.pin_decode();
        // The largest prompt group this budget funds, agreed by the ranks; its buffers are in the workspace below.
        let engine=engine.negotiate_prompt_group(budget)?;
        let resident=backend.accounted_bytes();let workspace=engine.workspace_reservation_bytes();
        let local=budget.saturating_sub(resident).saturating_sub(workspace);
        let pool=engine.negotiate_available_bytes(local)?;let reservation=engine.full_state_reservation_bytes();
        eprintln!("TEXT ADMISSION rank={rank} context={context} budget_bytes={budget} resident_bytes={resident} shared_workspace_bytes={workspace} session_pool_bytes={pool} full_session_bytes={reservation} cache_append_reservation_bytes={} index_append_reservation_bytes={} bf16_copy_bytes={} bf16_copy_matrices={} mhc_host_parameter_bytes={}",engine.cache_append_reservation_bytes(),engine.index_append_reservation_bytes(),spark_tp::resident_weights::BF16_COPY_BYTES.load(std::sync::atomic::Ordering::Relaxed),spark_tp::resident_weights::BF16_COPY_MATRICES.load(std::sync::atomic::Ordering::Relaxed),conductor.as_ref().map_or(0,|conductor|conductor.parameter_bytes()));
        eprintln!("TEXT PROMPT GROUP rank={rank} rows={} shared_workspace_bytes={workspace}",engine.prompt_group_rows());
        if spark_tp::row_split::admit_service_memory(resident,workspace,budget).is_err() || reservation>pool {
            return Err(format!("text service memory refusal: CONTEXT={context}; budget={budget}; resident={resident}; shared_workspace={workspace}; session_pool={pool}; full_session={reservation}; reduce CONTEXT"));
        }
        if !engine.controller() {return engine.worker();}
        let mut limits=spark_tp::row_split::limits(context,setting("FLASH_SERVE_OUTPUT_TOKENS",output_setting)?.unwrap_or(context.min(8192)))?;
        limits.next_mtp_depth=Some(engine.predictor_depth());limits.api_key=auth;
        limits.metadata=Some(spark_serve::metadata::Metadata::resident_text(&identity,world,wire_topology));
        limits.identity=Some(identity);
        limits.validate()?;
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:context,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
        let schedule=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:spark_tp::bounded_decoder::PROMPT_ROWS,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,
            draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        mailbox.require_success()?;
        let listener=TcpListener::bind(bind).map_err(|e|e.to_string())?;let stop=AtomicBool::new(false);
        let name=root.file_name().ok_or("checkpoint needs a directory name")?.to_string_lossy().to_ascii_lowercase();
        eprintln!("text_service qualification=unqualified diagnostic_option=--diagnostic-unqualified-text ready bind={bind} rank={rank} world={world} path=tensor_split context={context} resident_bytes={resident} session_bytes={reservation} shared_workspace_bytes={workspace} session_pool_bytes={pool} max_chats=8 drafting=false cache=false");
        let served=spark_serve::scheduler::serve_requests(listener,&engine,Some(&tokenizer),&limits,&cache,&schedule,&stop,&name,requests);
        let stopped=engine.shutdown();served.and(stopped)
            })();
            let drained=backend.synchronize();let result=result.and(drained);
            if result.is_err() {mailbox.fail();if let Some(conductor)=&conductor {conductor.fail();}}
            stop_wire.store(true,Ordering::Release);
            let joined=handle.join().map_err(|_|"text resident wire panicked".to_owned()).and_then(|v|v);
            let host_joined=host_worker.map_or(Ok(()),|worker|worker.finish());
            result.and(joined).and(host_joined)
        })
    })
}
fn setting(name:&str,value:Option<String>)->Result<Option<usize>> {value.map(|v|v.parse::<usize>().ok().filter(|&v|v>0).ok_or_else(||format!("{name} must be a positive byte or token count"))).transpose()}
#[cfg(test)]
mod tests {
    /// Weight readers inherit the parent's mask during load, so the text service places decode only after the
    /// weights and the optional predictor are resident.
    #[test] fn decode_is_placed_after_the_weights_load() {
        let source=include_str!("text.rs");let code=&source[..source.find("#[cfg(test)]").unwrap()];
        let place=code.find("placement.pin_decode()").unwrap();assert_eq!(code.matches("placement.pin_decode()").count(),1);
        assert!(place>code.find("Factory::new(").unwrap() && place>code.find(".with_predictor(").unwrap());
    }
}
