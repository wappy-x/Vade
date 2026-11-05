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
    let mut key=None;let mut mtp=None;let mut options=args[9..].iter();
    while let Some(option)=options.next() {match option.as_str() {
        "--diagnostic-unqualified-text"=>{},
        "--api-key-file" if key.is_none()=>key=Some(PathBuf::from(options.next().ok_or("--api-key-file needs a path")?)),
        "--mtp" if mtp.is_none()=>{let value=options.next().ok_or("--mtp needs a depth")?;if value!="0" {return Err("text service supports --mtp 0 only".into());}mtp=Some(0);},
        _=>return Err(format!("unsupported text service option: {option}")),
    }}
    super::settings::diagnostic_text_bind(bind)?;
    let auth=spark_serve::auth::ApiKey::load(key.as_deref(),false)?;
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(&root)?;
    let bound=spark_serve::text_checkpoint::CheckpointTextEdges::new(&checkpoint)?;
    let eos=tokenizer.checkpoint_eos_token_ids()?.ok_or("text service requires checkpoint EOS metadata")?;
    if eos.len()!=bound.edges().eos_tokens.len() || bound.edges().eos_tokens.iter().any(|id|!eos.contains(id) || !tokenizer.has_token(*id)) {return Err("service and tokenizer EOS sets differ".into());}
    let ptx=spark_gpu_probe::layer_ptx()?;
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let mut budget=free.saturating_sub((free/10).max(2<<30));
    if let Some(value)=setting("SEQPAR_MEMORY_BYTES",memory_setting)? {budget=budget.min(value);}
    if budget==0 {return Err("text service has no device memory budget".into());}
    let identity=spark_serve::contract::Identity::load(&root)?;
    let mut ring=Ring::connect(rank,world,listen,next)?;
    let wire_topology=ring.topology();let mut records=vec![0;world*80];let local=&mut records[rank*80..(rank+1)*80];
    local[..64].copy_from_slice(identity.fingerprint().as_bytes());local[64..72].copy_from_slice(&(context as u64).to_le_bytes());local[72..].copy_from_slice(&(budget as u64).to_le_bytes());
    ring.all_gather(&mut records,80)?;
    for record in records.chunks_exact(80) {
        if record[..72]!=records[..72] {return Err("text service rank metadata or context differs".into());}
        budget=budget.min(u64::from_le_bytes(record[72..].try_into().unwrap()).try_into().map_err(|_|"rank budget exceeds address space")?);
    }
    spark_gpu_probe::driver::with_backend(0,ptx,budget,|backend| {
        backend.require_resident_stream()?;
        let mut data_listen=listen;data_listen.set_port(listen.port().checked_add(2).ok_or("text data listen port exceeds range")?);
        let mut data_next=next;data_next.set_port(next.port().checked_add(2).ok_or("text data next port exceeds range")?);
        let data=Ring::connect(rank,world,data_listen,data_next)?;
        let split=Split::new(Device::with_ring(backend,Some(data)),ring)?;let engine=Factory::new(&checkpoint,split,context,eos.to_vec())?;
        let reservation=engine.session_reservation_bytes().ok_or("missing text service memory bound")?;
        // Factory::new has loaded every resident weight on this backend. Count what it holds.
        let resident=backend.allocated_bytes();spark_tp::row_split::admit_service_memory(resident,reservation,budget)?;
        if !engine.controller() {return engine.worker();}
        let mut limits=spark_tp::row_split::limits(context,setting("FLASH_SERVE_OUTPUT_TOKENS",output_setting)?.unwrap_or(context.min(16384)))?;
        limits.api_key=auth;
        limits.metadata=Some(spark_serve::metadata::Metadata::resident_text(&identity,world,wire_topology));
        limits.identity=Some(identity);
        limits.validate()?;
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:context,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
        let schedule=spark_serve::scheduler::Config {active:1,connections:32,prefill_chunk:1,foreground_rows:1,background_rows:1,active_bytes:reservation,session_budget_bytes:reservation,
            draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        let listener=TcpListener::bind(bind).map_err(|e|e.to_string())?;let stop=AtomicBool::new(false);
        let name=root.file_name().ok_or("checkpoint needs a directory name")?.to_string_lossy().to_ascii_lowercase();
        eprintln!("text_service qualification=unqualified diagnostic_option=--diagnostic-unqualified-text ready bind={bind} rank={rank} world={world} path=tensor_split context={context} resident_bytes={resident} session_bytes={reservation} drafting=false cache=false");
        let served=spark_serve::scheduler::serve_requests(listener,&engine,Some(&tokenizer),&limits,&cache,&schedule,&stop,&name,requests);
        let stopped=engine.shutdown();served.and(stopped)
    })
}
fn setting(name:&str,value:Option<String>)->Result<Option<usize>> {value.map(|v|v.parse::<usize>().ok().filter(|&v|v>0).ok_or_else(||format!("{name} must be a positive byte or token count"))).transpose()}
