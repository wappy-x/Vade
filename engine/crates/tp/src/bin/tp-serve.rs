//! Existing HTTP service backed by rank-zero TP session commands.
// Typed startup retains admission ownership; world2 remains refused pending state/service proof.
#[cfg(target_os="linux")]
struct HybridData(spark_seqpar::Ring);
#[cfg(target_os="linux")]
impl engine_fabric::exchange::Exchange for HybridData {
    fn gather_bytes(&mut self,v:&mut [u8],o:&[usize])->Result<(),String> { self.0.all_gather_bytes(v,o) }
    fn gather_f32(&mut self,v:&mut [f32],o:&[usize])->Result<(),String> { self.0.all_gather_f32(v,o) }
    fn reduce(&mut self,v:&mut [f32],s:&mut [f32])->Result<(),String> { self.0.all_reduce_f32(v,s) }
}
#[cfg(target_os="linux")]
impl Drop for HybridData { fn drop(&mut self) { self.0.abort(); } }
#[cfg(target_os="linux")]
fn start_hybrid_rank<T>(backend:&spark_gpu_probe::driver::Backend<'_>, data:spark_seqpar::Ring,
    plan:spark_engine::hybrid::tp::TransportPlan, rank:usize, active:usize, requested:Option<usize>,
    mut bounds:spark_serve::budget::HybridRankLedger,
    load:impl FnOnce(spark_engine::hybrid::tp::BoundedExchange<HybridData>,spark_serve::budget::HybridBudget,Vec<spark_serve::budget::HybridRankLedger>)->Result<T,String>)->Result<T,String> {
    use engine_fabric::exchange::Exchange;
    use spark_serve::budget::{HybridBudget,HybridRankLedger};
    if rank>=plan.ranks() { return Err("invalid hybrid startup rank".into()); }
    let mut data=HybridData(data);
    let prepared=plan.prepare(&mut data,|d,slot| { d.0.enable_mailbox(slot)?; Ok(d.0.mailbox_bytes()) },|d|d.0.seal_registrations())?;
    let ranks=plan.ranks();
    prepared.before_weights(data,|e,mailbox,host| {
        let local=(|| {
            let free=backend.available_memory_bytes()?.min(backend.device_memory_info()?.0);
            let owned=backend.accounted_bytes();
            let fixed=owned.checked_add(mailbox).ok_or("hybrid measured owner overflow")?;
            let safety=(free/10).max(2<<30);
            bounds.ceiling=bounds.ceiling.min(free.saturating_sub(safety).checked_add(fixed).ok_or("hybrid measured ceiling overflow")?);
            bounds.shared=bounds.shared.checked_add(host).and_then(|n|n.checked_add(owned)).ok_or("hybrid shared owner overflow")?;
            bounds.transport=mailbox; bounds.measured_fixed=fixed;
            eprintln!("hybrid bootstrap rank={rank} free_bytes={free} safety_bytes={safety} owned_bytes={owned} mailbox_bytes={mailbox} host_bound_bytes={host}");
            Ok::<_,String>(bounds)
        })();
        if let Err(error)=&local { eprintln!("hybrid startup rank={rank}: {error}"); }
        let mut wire=vec![0u8;ranks*112]; let own=&mut wire[rank*112..(rank+1)*112];
        let values=local.as_ref().map(|l|l.values()).unwrap_or([0;8]);
        for (out,value) in own.chunks_exact_mut(8).zip(values.into_iter().chain([active,requested.unwrap_or(0),usize::from(requested.is_some()),usize::from(local.is_err()),plan.wire_bytes(),host])) { out.copy_from_slice(&(value as u64).to_le_bytes()); }
        e.gather_bytes(&mut wire,&(0..=ranks).map(|r|r*112).collect::<Vec<_>>())?;
        let mut ledgers=Vec::new();
        for (r,bytes) in wire.chunks_exact(112).enumerate() {
            let v=bytes.chunks_exact(8).map(|b|usize::try_from(u64::from_le_bytes(b.try_into().unwrap())).map_err(|_|"hybrid ledger extent")).collect::<Result<Vec<_>,_>>()?;
            if v[11]!=0 { return Err(format!("hybrid rank {r} measurement failed before weights")); }
            if v[8..11]!=[active,requested.unwrap_or(0),usize::from(requested.is_some())] { return Err("hybrid ranks disagree on admission settings".into()); }
            if v[12..14]!=[plan.wire_bytes(),host] { return Err("hybrid ranks disagree on transport envelope".into()); }
            let ledger=HybridRankLedger::from_values(v[..8].try_into().unwrap());
            eprintln!("hybrid startup rank={r} ledger={ledger:?} transport_sealed=true"); ledgers.push(ledger);
        }
        let pool=HybridBudget::for_ranks(&ledgers,active,requested)?; let ledger=local?;
        let device=ledger.resident.checked_add(ledger.shared).and_then(|n|n.checked_sub(host));
        let ceiling=device.ok_or_else(||"hybrid device ceiling overflow".to_owned()).and_then(|n|backend.set_allocation_budget(n));
        let mut votes=vec![0;ranks]; votes[rank]=u8::from(ceiling.is_err()); e.gather_bytes(&mut votes,&(0..=ranks).collect::<Vec<_>>())?;
        if votes.contains(&1) { return Err("hybrid rank allocation ceiling refused before weights".into()); }
        Ok((pool,ledgers))
    },|e,(pool,ledgers)| {
        // Loading owns resident/shared bytes only. Publish readiness after return.
        let ledger=ledgers[rank]; let loaded=load(e,pool,ledgers)?;
        let device=ledger.resident.checked_add(ledger.shared).and_then(|n|n.checked_sub(plan.host_bytes()))
            .and_then(|n|n.checked_add(pool.sessions)).ok_or("hybrid final device ceiling overflow")?;
        backend.set_allocation_budget(device)?;
        eprintln!("hybrid loaded rank={rank} device_bytes={} device_peak_bytes={} accounted_peak_bytes={} host_heap_peak_bytes={} pool_bytes={} ledger={ledger:?}",
            backend.allocated_bytes(),backend.peak_allocated_bytes(),backend.peak_accounted_bytes(),spark_tp::runtime::heap_peak(),pool.sessions);
        Ok(loaded)
    })
}
#[cfg(target_os="linux")]
fn run_typed(a:&[String],ptx:&std::ffi::CStr,root:&std::path::Path,mut control:spark_tp::transport::Ring,
    mut limits:spark_serve::Limits,active:usize,count:usize,bind:std::net::SocketAddr,server_start:std::time::SystemTime)->Result<(),String> {
    use engine_model::{plan::ModelPlan,split::{SplitPlan,SplitRequest}};
    use spark_engine::hybrid::{Model,tp::{Binding,Group}};
    let (rank,world)=(control.rank(),control.world()); let context=limits.context_tokens;
    let (plan,index,headers)=ModelPlan::inspect_parts(root).map_err(|e|e.to_string())?;
    if !spark_engine::hybrid::plan::uses_typed_loader(&plan) { return Err("TP hybrid requires the typed FP8 cartridge".into()); }
    let depth=match std::env::var("MTP").as_deref() { Err(_)=>0,Ok(n)=>n.parse::<usize>().ok().filter(|n|*n<=4).ok_or("hybrid MTP must be 0..=4")? }; let mtp=depth!=0;
    let split=SplitPlan::new(&plan.shape,&SplitRequest {ranks:world as u32,memory:vec![u64::MAX;world],reserve:vec![0;world],
        context:context as u64,sequences:1,cache_bytes:2,state_bytes:4,draft:mtp,expert_rows:false}).map_err(|e|e.to_string())?;
    let members=(1..=world as u64).collect::<Vec<_>>(); let incarnation=spark_tp::runtime::run_hash().max(1);
    let binding=Binding::new(&plan,&split,rank)?;
    let rows=if mtp {spark_engine::hybrid::decode::MAX_VERIFY_ROWS}else{spark_engine::hybrid::decode::MAX_ROWS};
    let selected=plan.shape.layers.iter().chain(&plan.shape.draft).filter_map(|l|if let engine_model::shapes::Ffn::Experts(e)=l.ffn{Some(e.selected as usize)}else{None}).max().unwrap_or(0);
    let mut setup=Group::setup_words(&plan,&split,&members,incarnation); setup.extend([context as u64,active as u64,depth as u64]);
    let command_words=setup.len().max(1+2*rows*selected).max(
        spark_tp::service::HybridProtocol::new(1,&plan,&split,rank as u32,&members,incarnation)?.command_words(context.checked_add(1).ok_or("hybrid payload overflow")?)?);
    let transport=binding.transport_plan(rows,command_words)?;
    let [resident,shared,request]=binding.rank_reservations(&index,&headers,context,mtp)?;
    let endpoint=|text:&str|->Result<std::net::SocketAddr,String> { let mut address=text.parse::<std::net::SocketAddr>().map_err(|e|e.to_string())?;
        address.set_port(address.port().checked_add(1024).ok_or("hybrid data port overflow")?);Ok(address) };
    let id=format!("{:016x}:typed",incarnation);
    let data=if control.topology()=="path" {spark_seqpar::Ring::connect_path(rank,world,endpoint(&a[2])?,endpoint(&a[3])?,&id)?}
        else {spark_seqpar::Ring::connect(rank,world,endpoint(&a[2])?,endpoint(&a[3])?,&id)?};
    let ceiling=spark_seqpar::memory_budget()?;
    let result=spark_gpu_probe::driver::with_backend(0,ptx,ceiling,|b| {
        let ledger=spark_serve::budget::HybridRankLedger {ceiling,resident,shared,request,transport:0,retained:0,in_flight:0,measured_fixed:0};
        let (model,pool,ledgers)=start_hybrid_rank(b,data,transport,rank,active,None,ledger,|mut e,pool,ledgers| {
            Binding::new(&plan,&split,rank)?.agree(&mut e,&setup,&[])?;
            let model=Model::load_generic_rank(b,root,context,mtp,plan,&index,&headers,split,rank,&members,incarnation,e)?;
            Ok((model,pool,ledgers))
        })?;
        let tokenizer=engine_tokenizer::Tokenizer::from_directory(root).map_err(|e|e.to_string())?;
        let engine=spark_serve::hybrid::Hybrid::new(&model,tokenizer.eos_token_id()).with_draft_depth(depth)?;
        let workspace=engine.reserve_workspace();
        if model.rank_group()?.agree(&[0x5354415254],&[!workspace.as_ref().is_ok_and(|&n|n<=shared)])?[0] { return Err("hybrid workspace exceeded admission".into()); }
        workspace?; control.barrier()?;
        limits.default_sampling=Some(spark_engine::sample::Sampling::from_checkpoint(root)?);
        limits.identity=Some(spark_serve::contract::Identity::load(root)?);
        limits.metadata=Some(spark_serve::metadata::Metadata::load(root,ptx,depth,world,control.topology(),server_start)?);
        let controller=spark_tp::service::typed::Controller::new(&model,&engine,control,&ledgers)?;
        if rank!=0 { return controller.worker(); }
        let listener=std::net::TcpListener::bind(bind).map_err(|e|e.to_string())?;
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:pool.sessions,tokens_per_entry:context,logits_per_entry:model.config().vocab as usize,idle_ttl:std::time::Duration::from_secs(300)};
        let schedule=spark_serve::scheduler::Config {active,connections:32.max(active),prefill_chunk:512,foreground_rows:8,background_rows:16,
            active_bytes:pool.sessions,session_budget_bytes:pool.sessions,draft:if mtp {spark_serve::protocol::DraftMode::Strict}else{spark_serve::protocol::DraftMode::Plain},mtp_depth:depth,
            draft_global_bank:false,prefix_cache:true,sharing:spark_serve::scheduler::Sharing::default()};
        eprintln!("TP typed service ready {bind} world={world} context={context} ledgers={ledgers:?}");
        let served=spark_serve::scheduler::serve_requests(listener,&controller,Some(&tokenizer),&limits,&cache,&schedule,controller.shutdown_requested(),&root.file_name().unwrap_or_default().to_string_lossy(),count);
        served.and(controller.shutdown())
    });
    result
}
#[cfg(target_os="linux")]
fn run()->Result<(),String>{
    use std::{ffi::CString,path::PathBuf,net::{TcpListener,SocketAddr}};
    use spark_tp::{model::Model,service::{Engine,worker_with_sequence,SEQUENCE_THRESHOLD},transport::Ring};
    spark_gpu_probe::teardown::install()?;
    let server_start=std::time::SystemTime::now();
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    if a.len()!=9{return Err("usage: tp-serve RANK WORLD LISTEN NEXT PTX MODEL CONTEXT BIND REQUESTS (0=continuous)".into());}
    let rank=a[0].parse::<usize>().map_err(|e|e.to_string())?;let world=a[1].parse::<usize>().map_err(|e|e.to_string())?;
    let context=a[6].parse::<usize>().map_err(|e|e.to_string())?;
    let count=a[8].parse::<usize>().map_err(|e|e.to_string())?;
    let bind=a[7].parse::<SocketAddr>().map_err(|e|e.to_string())?;
    if !bind.ip().is_loopback(){return Err("TP service binds loopback only".into());}
    let active=std::env::var("TP_SERVE_MAX_ACTIVE").unwrap_or_else(|_|"2".into()).parse::<usize>().map_err(|_|"invalid TP_SERVE_MAX_ACTIVE")?;
    if !(2..=32).contains(&active){return Err("TP_SERVE_MAX_ACTIVE must be in 2..=32".into());}
    let mut limits=spark_serve::Limits::default();limits.context_tokens=context;limits.input_tokens=context.saturating_sub(1);limits.output_tokens=context.min(256);
    if let Ok(value)=std::env::var("TP_SERVE_GENERATION_TIMEOUT_MS") {
        limits.generation_timeout=std::time::Duration::from_millis(value.parse().map_err(|_|"invalid TP_SERVE_GENERATION_TIMEOUT_MS")?);
    }
    limits.validate()?;
    let ptx=CString::new(std::fs::read(&a[4]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let root=if let Some(p)=a[5].strip_prefix("~/"){PathBuf::from(std::env::var("HOME").map_err(|e|e.to_string())?).join(p)}else{PathBuf::from(&a[5])};
    let mut ring=Ring::connect(rank,world,a[2].parse().map_err(|_|"listen")?,a[3].parse().map_err(|_|"next")?)?;
    spark_tp::runtime::verify_artifacts(&mut ring, &ptx, &root)?;
    if engine_model::cartridge::Kind::detect(&root).map_err(|e|e.to_string())?==engine_model::cartridge::Kind::Qwen3_5 {
        return run_typed(&a,&ptx,&root,ring,limits,active,count,bind,server_start);
    }
    let config=engine_model::qwen3::Config::open(&root.join("config.json")).map_err(|e|e.to_string())?;
    // Rank zero owns the largest shard, so all ranks make identical admission decisions.
    let shard_session=Model::service_session_bytes(&config,0,world,context);
    let shard_resident=Model::device_budget(&config,0,world,context)?
        -Model::shard_session_bytes(&config,0,world,context);
    let shard_budget=shard_resident
        .checked_add(shard_session.checked_mul(active+1).ok_or("session budget overflow")?).ok_or("device budget overflow")?;
    let available=spark_seqpar::memory_budget()?;
    // Every rank must select the same path, including on unequal free memory, and the same
    // prompt path (SPARK_PROFILE changes the link step sequence).
    let profile=Model::profile_requested();
    let mut budgets=vec![0u8;world*16];budgets[rank*16..rank*16+8].copy_from_slice(&(available as u64).to_le_bytes());
    budgets[rank*16+8]=profile as u8;
    ring.all_gather(&mut budgets,16)?;
    let available=budgets.chunks_exact(16).map(|b|u64::from_le_bytes(b[..8].try_into().unwrap()) as usize).min().unwrap();
    if budgets.chunks_exact(16).any(|b|b[8]!=profile as u8){return Err("SPARK_PROFILE must be the same on every rank".into());}
    let cap=context.div_ceil(16)*16;
    let (h,q,k)=(config.hidden_size as usize,(config.attention_heads*config.head_dim) as usize,(config.kv_heads*config.head_dim) as usize);
    let full_resident=Model::device_budget(&config,0,1,context)?-Model::shard_session_bytes(&config,0,1,context)
        +2*config.vocab_size as usize*h;
    let sequence_session=8*config.layers as usize*cap*k+4*cap*(h+q+2*k)
        +4*cap.min(4096)*(8*h+4*q+4*k+2*config.intermediate_size as usize);
    // Sequence-parallel prompt prefill hands off an FP32 KV cache, so its answers differ from one
    // Spark's BF16 KV: opt-in. Without it hybrid sessions keep BF16 KV and prefill on the shards,
    // so they are charged as shard sessions.
    let sequence_prefill=match std::env::var("TP_SERVE_SEQUENCE_PREFILL").as_deref(){
        Ok("1")=>true,Ok("0")|Err(_)=>false,_=>return Err("TP_SERVE_SEQUENCE_PREFILL must be 0 or 1".into()),
    };
    let hybrid_session=if sequence_prefill{Model::shard_session_bytes(&config,0,world,context).saturating_add(sequence_session)}else{shard_session};
    let hybrid_budget=full_resident.saturating_add(hybrid_session.saturating_mul(active+1));
    // Hybrid pays off only through that prompt split (its chats never share a pass), so auto needs it.
    let hybrid=match std::env::var("TP_SERVE_HYBRID").as_deref(){
        Ok("0")=>false,Ok("1")=>true,Ok("auto")|Err(_)=>sequence_prefill&&context>=SEQUENCE_THRESHOLD&&hybrid_budget<=available,
        _=>return Err("TP_SERVE_HYBRID must be 0, 1, or auto".into()),
    };
    let device_budget=if hybrid{hybrid_budget}else{shard_budget};
    if device_budget>available{return Err(spark_gpu_probe::admission::refusal(device_budget as u128, available, &format!("weights/rank and fixed workspace {} bytes; per-session {} bytes x {} sessions (one cache slot included); CONTEXT={context}; OUTPUT={}", if hybrid {full_resident} else {shard_resident}, if hybrid {hybrid_session} else {shard_session}, active+1, limits.output_tokens), "the lowest rank's SEQPAR_MEMORY_BYTES or 90% of /proc/meminfo MemAvailable", "positional CONTEXT or TP_SERVE_MAX_ACTIVE; TP_SERVE_HYBRID=0 reduces replication"));}
    let sequence_prefill=hybrid&&sequence_prefill;
    if sequence_prefill&&std::env::var("SEQPAR_SERIAL_COMPUTE").as_deref()==Ok("1"){return Err("hybrid service requires concurrent ranks; unset SEQPAR_SERIAL_COMPUTE".into());}
    let sequence_ring=if sequence_prefill&&world>1 {
        let endpoint=|text:&str|->Result<SocketAddr,String>{
            let mut address=text.parse::<SocketAddr>().map_err(|e|e.to_string())?;
            address.set_port(address.port().checked_add(1024).ok_or("sequence port overflow")?);Ok(address)
        };
        let id=std::env::var("TP_RUN_ID").map_err(|_|"hybrid service needs TP_RUN_ID for multi-rank runs")?;
        let config=std::fs::read(root.join("config.json")).map_err(|e|format!("read model config: {e}"))?;
        let signature=(context as u64).to_le_bytes().into_iter().chain(config)
            .fold(0xcbf29ce484222325u64,|hash,byte|(hash^u64::from(byte)).wrapping_mul(0x100000001b3));
        let id=format!("{id}:serve:{signature:016x}");
        Some(if ring.topology()=="path"{spark_seqpar::Ring::connect_path(rank,world,endpoint(&a[2])?,endpoint(&a[3])?,&id)?}
            else{spark_seqpar::Ring::connect(rank,world,endpoint(&a[2])?,endpoint(&a[3])?,&id)?})
    }else{None};
    spark_gpu_probe::driver::with_backend(0,&ptx,device_budget,|b|{
        let topology=ring.topology();
        let full=if hybrid{
            let mut model=spark_engine::qwen::Model::load_with_kv_budget(b,&root,context,device_budget)?;
            model.prepare_batch_weights()?;Some(model)
        }else{None};
        let mut m=if let Some(full)=&full{Model::from_full(full,rank,world)?}else{Model::load(b,&root,rank,world,context)?};
        if sequence_prefill{m.enable_sequence_prefill();}
        let bound=m.session_reservation_bytes();
        if m.resident_bytes.saturating_add((active+1).saturating_mul(bound))>device_budget{return Err(spark_gpu_probe::admission::refusal(m.resident_bytes as u128 + (active as u128+1)*bound as u128, device_budget, &format!("weights/rank {}; per-session {bound} bytes x {} sessions (one cache slot included); CONTEXT={context}; OUTPUT={}", m.resident_bytes, active+1, limits.output_tokens), "the negotiated per-rank device ceiling", "positional CONTEXT or TP_SERVE_MAX_ACTIVE"));}
        ring.barrier()?;
        eprintln!("memory rank={rank} phase=loaded model_bytes={} device_bytes={} device_peak_bytes={} session_reservation_bytes={bound}",m.resident_bytes,b.allocated_bytes(),b.peak_allocated_bytes());
        let result=if rank!=0{worker_with_sequence(&m,ring,sequence_ring)}else{(||{
            let engine=Engine::new_with_sequence(&m,ring,sequence_ring)?;
            let tokenizer=engine_tokenizer::Tokenizer::from_directory(&root).map_err(|e|e.to_string())?;
            let listener=TcpListener::bind(bind).map_err(|e|e.to_string())?;
            // Cache accounting includes token/logit vectors alongside the session.
            let retained=bound+4*(context+m.config().vocab_size as usize)
                +160*(1+context/64)+96*2*config.layers as usize*(1+context/512);
            let cache=spark_serve::cache::CacheLimits{entries:1,retained_bytes:retained,tokens_per_entry:context,logits_per_entry:m.config().vocab_size as usize,idle_ttl:std::time::Duration::from_secs(300)};
            let schedule=spark_serve::scheduler::Config{active,connections:32.max(active),prefill_chunk:512,foreground_rows:8,background_rows:16,active_bytes:active*bound,session_budget_bytes:active*bound+retained,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:true,sharing:spark_serve::scheduler::Sharing::default()};
            eprintln!("TP service ready {bind} world={world} context={context} max_active={active} seqpar_threshold={} model_bytes_per_rank={}",if sequence_prefill{SEQUENCE_THRESHOLD}else{0},m.resident_bytes);
            let model=root.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
            limits.metadata=Some(spark_serve::metadata::Metadata::load(&root,&ptx,0,world,topology,server_start)?);
            let served=spark_serve::scheduler::serve_requests(listener,&engine,Some(&tokenizer),&limits,&cache,&schedule,engine.shutdown_requested(),&model,count);
            let stopped=engine.shutdown();
            served.and(stopped)
        })()};
        eprintln!("memory rank={rank} phase=exit model_bytes={} device_bytes={} device_peak_bytes={} pinned_bytes={} host_heap_peak_bytes={} process_hwm_bytes={}",m.resident_bytes,b.allocated_bytes(),b.peak_allocated_bytes(),b.pinned_host_bytes(),spark_tp::runtime::heap_peak(),spark_tp::runtime::host_memory().1);
        result
    })
}
fn main(){
    #[cfg(target_os="linux")]
    if let Err(e)=run(){eprintln!("FAIL {e}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    {eprintln!("requires Linux CUDA driver");std::process::exit(1);}
}
#[global_allocator]
static ALLOCATOR:spark_tp::runtime::TrackedAllocator=spark_tp::runtime::TrackedAllocator;
