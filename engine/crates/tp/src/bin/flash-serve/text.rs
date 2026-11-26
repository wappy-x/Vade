//! Bounded text service on the shared HTTP scheduler and complete-row transport.
use engine_model::plan::{admission::CheckpointAdmission,layer::Topology};
use spark_serve::{Engine,Tokenizer};
use spark_tp::{Result,row_split::{Factory,Split},bounded_decoder::Device,startup,transport::Ring};
use std::{net::{SocketAddr,TcpListener},path::Path,sync::atomic::AtomicBool,time::Duration};
#[path="boundary_gate.rs"] mod boundary_gate;
/// `started`: when the process started. The caller gives a failure to `startup::failed`, which prints the rank and the step of a failed start.
pub fn run(args:&[String],checkpoint:CheckpointAdmission,root:&Path,output_setting:Option<String>,memory_setting:Option<String>,started:std::time::Instant)->Result<()> {
    let diagnostic=super::settings::diagnostic_text(args.get(9..).unwrap_or_default())?;
    if !diagnostic {checkpoint.description().require_resident_service().map_err(|e|format!("text service refused: {e}; source precision and zero-sign policy require owner approval"))?;}
    if args.len()<9 {return Err(format!("text service needs RANK WORLD LISTEN NEXT PTX MODEL CONTEXT BIND REQUESTS; {}",super::settings::TEXT_OPTIONS));}
    let rank=args[0].parse::<usize>().map_err(|e|e.to_string())?;let world=args[1].parse::<usize>().map_err(|e|e.to_string())?;
    startup::begin(rank,started,"header_admission");
    let topology=Topology {rank:rank.try_into().map_err(|_|"rank exceeds topology")?,parts:world.try_into().map_err(|_|"world exceeds topology")?};topology.validate().map_err(|e|e.to_string())?;
    let listen=args[2].parse::<SocketAddr>().map_err(|e|e.to_string())?;let next=args[3].parse::<SocketAddr>().map_err(|e|e.to_string())?;
    let context=args[6].parse::<usize>().map_err(|e|e.to_string())?;let bind=args[7].parse::<SocketAddr>().map_err(|e|e.to_string())?;
    let requests=args[8].parse::<usize>().map_err(|e|e.to_string())?;
    let (gate,options)=super::settings::boundary_gate_options(&args[9..])?;
    let (key,mtp,trace,keep,cohort_chats)=super::settings::text_options(&options)?;
    if context>u32::MAX as usize {return Err("text context exceeds token position range".into());}
    if gate.is_some() && (!diagnostic || mtp!=Some(3)) {return Err("boundary gate requires --diagnostic-unqualified-text and --mtp 3".into());}
    // One line for a hand start that sets a cache size of 0 and still keeps prompts. Nothing else follows from it.
    if let Some(notice)=super::settings::text_cache_size_notice(keep,spark_tp::flash_pool::Request::from_env().ok().and_then(|request|request.cache_bytes)) {eprintln!("TEXT PROMPT KEEP rank={rank} {notice}");}
    super::settings::diagnostic_text_bind(bind)?;
    let auth=spark_serve::auth::ApiKey::load(key.as_deref(),false)?;
    // The tokenizer, the identity hashes and the memory query read different things, so they run at the same time.
    startup::step("tokenizer_identity_memory");
    let (tokenizer,identity,memory)=std::thread::scope(|scope| {
        let tokenizer=scope.spawn(||spark_serve::tokenizer::ModelTokenizer::from_directory(root));
        let identity=scope.spawn(||spark_serve::contract::Identity::load(root));
        let memory=spark_gpu_probe::driver::device_memory_info(0);
        (tokenizer.join().map_err(|_|"tokenizer reader stopped".to_owned()).and_then(|loaded|loaded),identity.join().map_err(|_|"identity reader stopped".to_owned()).and_then(|loaded|loaded),memory)
    });
    let mut tokenizer=tokenizer?;
    startup::step("text_edges");
    let bound=spark_serve::text_checkpoint::CheckpointTextEdges::new(&checkpoint)?;
    let eos=tokenizer.checkpoint_eos_token_ids()?.ok_or("text service requires checkpoint EOS metadata")?.to_vec();
    if eos.len()!=bound.edges().eos_tokens.len() || bound.edges().eos_tokens.iter().any(|id|!eos.contains(id) || !tokenizer.has_token(*id)) {return Err("service and tokenizer EOS sets differ".into());}
    let ptx=spark_gpu_probe::layer_ptx()?;
    let (_,total)=memory?;let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    let mut budget=free.saturating_sub((free/10).max(2<<30));
    if let Some(value)=setting("SEQPAR_MEMORY_BYTES",memory_setting)? {budget=budget.min(value);}
    if budget==0 {return Err("text service has no device memory budget".into());}
    let identity=identity?;
    startup::step("rank_connect");
    let _cpu_pin=spark_tp::runtime::defer_cpu_pin();
    let mut ring=Ring::connect(rank,world,listen,next)?;
    // The small commands of a step go rank to rank on RDMA mailboxes where the ring relays them (`Ring::relays_commands`:
    // a path of COMMAND_RELAY_RANKS or more ranks with RDMA links), as a pair sends them on its mailbox. Without the relay
    // every hop is a read of the control stream. Every rank asks the same question at this same point, so all open the
    // relay or none does. A pair's ring answers no.
    #[cfg(target_os="linux")]
    if ring.relays_commands() {ring.open_command_relay()?;eprintln!("TEXT COMMAND RELAY transport=rdma_mailbox");}
    startup::step("rank_records");
    let wire_topology=ring.topology();let mut records=vec![0;world*80];let local=&mut records[rank*80..(rank+1)*80];
    // The top bit of the context word says this rank keeps prompts, so ranks with different choices refuse each other:
    // different options, or one build of before the default (off without an option) and one of after it (on).
    // Bit 62 goes with it since a resume starts with the readiness word (`Factory::kept_ready`): a build of before
    // that word keeps prompts with bit 62 clear, so it and this build refuse each other here, not at a first resume.
    // Bit 61 identifies the lossless BF16 prompt packet format. A mixed build
    // must fail here, before either rank posts a differently sized collective.
    // Bits 57 through 60 carry the agreed maximum chats in a prompt cohort.
    // Bit 56 identifies the native fold and mixed router-stage packet layouts.
    let context_word=(context as u64|u64::from(keep)<<63|u64::from(keep)<<62|1<<61)|(cohort_chats as u64)<<57|1<<56;
    local[..64].copy_from_slice(identity.fingerprint().as_bytes());local[64..72].copy_from_slice(&context_word.to_le_bytes());local[72..].copy_from_slice(&(budget as u64).to_le_bytes());
    ring.all_gather(&mut records,80)?;
    for record in records.as_chunks::<80>().0 {
        if record[..72]!=records[..72] {return Err("text service rank metadata, wire format, context, prompt cohort limit or prompt keep choice differs: start every rank from one build with the same options (prompts are kept unless --no-prompt-keep is given)".into());}
        budget=budget.min(u64::from_le_bytes(record[72..].try_into().unwrap()).try_into().map_err(|_|"rank budget exceeds address space")?);
    }
    // The driver compiles a kernel image that is new to this box, one time, for about 40 s. A warm load takes 0.35 s. The log names the wait after 3 s, so nobody stops the rank in the middle of a compile.
    startup::step("gpu_context_and_kernel_image");
    let image=std::sync::Arc::new(AtomicBool::new(false));
    {let image=image.clone();std::thread::spawn(move|| {std::thread::sleep(Duration::from_secs(3));if !image.load(std::sync::atomic::Ordering::Acquire) {eprintln!("STARTUP rank={rank} the GPU context and the kernel image are not up after 3 s. The driver compiles a kernel image that is new to this box one time, for about 40 s: do not stop this rank");}});}
    spark_gpu_probe::driver::with_backend(0,ptx,budget,|backend| {
        image.store(true,std::sync::atomic::Ordering::Release);
        // Each rank connect below has a 30 s limit. A rank that compiles a new kernel image is still in the line above, so the
        // ranks meet here first, on the link they already hold: this barrier waits up to 180 s and ends at once if the other rank stops.
        startup::step("kernel_image_wait");
        ring.barrier()?;
        if let Some(waited)=startup::waited().filter(|&seconds|seconds>=1.0) {eprintln!("STARTUP rank={rank} waited {waited:.1} s for the kernel image of the other rank");}
        startup::step("resident_stream");
        backend.require_resident_stream()?;
        startup::step("data_connect");
        let mut data_listen=listen;data_listen.set_port(listen.port().checked_add(2).ok_or("text data listen port exceeds range")?);
        let mut data_next=next;data_next.set_port(next.port().checked_add(2).ok_or("text data next port exceeds range")?);
        let data=Ring::connect(rank,world,data_listen,data_next)?;
        startup::step("mailbox");
        // One receive channel for each other rank, with room for a message of 4 MiB (`tp_route::Channels`). For two ranks
        // this IS the pair's mailbox: `tp_mailbox(4<<20)` is this same call with the links of a pair.
        let mailbox=backend.tp_mailbox_routed(spark_gpu_probe::tp_route::Channels::new(&spark_gpu_probe::tp_route::Links::new(world,false)?,(4<<20)/4)?)?;
        startup::step("mhc_conductor");
        // The host mHC lane takes the rows of a batch step too (one row a core).
        let conductor=spark_tp::host_conductor::Conductor::with_rows(backend,&checkpoint,3,spark_tp::bounded_decoder::BATCH_ROWS)?;
        startup::step("placement");
        let mut placement=spark_tp::thread_affinity::Placement::new(&mut ring,conductor.as_ref().map_or(0,|conductor|conductor.cores()))?;
        let physical=Topology {rank:topology.rank,parts:mailbox.layout().world.try_into().map_err(|_|"mailbox topology exceeds ABI")?};let compatible=topology.eq(&physical);
        if !compatible {return Err("diagnostic text service: the mailbox layout and the session topology differ".into());}
        let port=|mut address:SocketAddr|->Result<SocketAddr> {address.set_port(address.port().checked_add(1).ok_or("text wire port exceeds range")?);Ok(address)};
        use spark_gpu_probe::tp_route::{Links,Side};
        let links=Links::new(world,false)?;
        let left=links.neighbor(rank,Side::Left).map(|_|port(data_listen)).transpose()?;
        let right=links.neighbor(rank,Side::Right).map(|_|port(data_next)).transpose()?;let lead=left.is_none();
        startup::step("wire_connect");
        // `connect_peers` gives the link this rank connected (toward the higher rank), then the link it accepted.
        let (outgoing,incoming)=spark_seqpar::control::connect_peers(left,right,std::time::Instant::now()+Duration::from_secs(300))?;
        startup::step("wire_link");
        // The links this rank has, left then right. `Wire::bind` gives a pair its write wire; on a longer path the wire
        // thread writes this rank's messages to both neighbors and relays the others on.
        let end=|link:Option<std::net::TcpStream>|link.as_ref().map(spark_tp::tensor_wire::WriteLink::new).transpose();
        let wire=spark_tp::tensor_wire::Wire::Relay([end(incoming)?,end(outgoing)?],links,rank);
        let worker=wire.bind(mailbox.take_host().ok_or("text mailbox endpoint is taken")?,lead,mailbox.layout())?;
        use std::sync::atomic::Ordering;
        let (stop_wire,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
        std::thread::scope(|scope| {
            startup::step("wire_attach");
            let workers=placement.workers();
            let host_worker=conductor.as_ref().map(|conductor|conductor.spawn(scope)).transpose()?;
            let handle=spark_tp::tensor_wire::spawn_on(scope,worker,&stop_wire,&attached,&failed,placement.wire_mask())?;
            drop(workers);
            let result=(|| {
                spark_tp::tensor_wire::wait_attached(&attached,&handle)?;
        startup::step("device_setup");
        let compute=Device::with_mailbox(backend,data,false,&mailbox)?.with_serving_trace(trace);
        let compute=match &conductor {Some(conductor)=>compute.with_mhc_host(conductor.link())?,None=>compute};
        // Fused boundaries use the admitted row loop or the ordered phases of swept wider cohorts.
        let compute=compute.with_mhc_fused()?;
        // The decoder ends this step and starts the step `weights` when its host parameters are read.
        let split=Split::new(compute,ring)?;startup::step("decoder_parameters");let engine=Factory::new(&checkpoint,split,context,eos)?;
        startup::step("predictor");
        // The predictor layer is loaded only for --mtp 1..3, before admission counts the resident bytes.
        let engine=match mtp {Some(depth) if depth>0=>engine.with_predictor(depth)?,_=>engine};
        // Prompts are kept unless --no-prompt-keep is given: the scheduler's prompt cache may then fork this service's
        // sessions (row_split `Session::fork`).
        let engine=if keep {engine.with_prompt_keep()} else {engine};
        // Load readers keep the initial mask; decode takes its mask once the weights are resident.
        placement.pin_decode();
        startup::step("prompt_group");
        // The largest prompt group this budget funds, agreed by the ranks; its buffers are in the workspace below.
        // Before the group is chosen: a server whose latent layers are all absorbed is not charged the expanded rows it never makes.
        let (engine,floor,predicted)=engine.with_absorbed_charges()?.with_prompt_group_and_memo(budget)?;
        let engine=engine.with_prompt_cohort_chats(cohort_chats)?;
        startup::step("memory_admission");
        let resident=backend.accounted_bytes();let workspace=engine.workspace_reservation_bytes();
        let local=budget.saturating_sub(resident).saturating_sub(workspace);
        let pool=engine.negotiate_available_bytes(local)?;let reservation=engine.full_state_reservation_bytes();
        eprintln!("TEXT ADMISSION rank={rank} context={context} budget_bytes={budget} resident_bytes={resident} shared_workspace_bytes={workspace} session_pool_bytes={pool} full_session_bytes={reservation} cache_append_reservation_bytes={} index_append_reservation_bytes={} bf16_copy_bytes={} bf16_copy_matrices={} mhc_host_parameter_bytes={}",engine.cache_append_reservation_bytes(),engine.index_append_reservation_bytes(),spark_tp::resident_weights::BF16_COPY_BYTES.load(std::sync::atomic::Ordering::Relaxed),spark_tp::resident_weights::BF16_COPY_MATRICES.load(std::sync::atomic::Ordering::Relaxed),conductor.as_ref().map_or(0,|conductor|conductor.parameter_bytes()));
        eprintln!("TEXT PROMPT GROUP rank={rank} rows={} shared_workspace_bytes={workspace}",engine.prompt_group_rows());
        eprintln!("TEXT PROMPT GROUP FLOOR rank={rank} rows={} floor_bytes={floor} predicted_pool_bytes={predicted} local_pool_bytes={local}",engine.prompt_group_rows());
        if local!=predicted {eprintln!("TEXT PROMPT GROUP FLOOR WARNING predicted and admitted pool differ rank={rank} predicted_pool_bytes={predicted} local_pool_bytes={local}");}
        eprintln!("TEXT PROMPT COHORT CAP rank={rank} chats={cohort_chats}");
        if cohort_chats<spark_tp::row_split::LIVE_SESSIONS {
            eprintln!("TEXT PROMPT COHORT WARNING rank={rank} --prompt-cohort-chats={cohort_chats} slots={} disables whole prompt groups",spark_tp::row_split::LIVE_SESSIONS);
        }
        match engine.prompt_cohort_growth() {
            Ok((before,at,tail))=>eprintln!("TEXT PROMPT COHORT BOUND rank={rank} row_growth_bytes={} tail_bytes={tail}",at as i128-before as i128),
            Err(error)=>eprintln!("TEXT PROMPT COHORT BOUND rank={rank} error={error}"),
        }
        if spark_tp::row_split::admit_service_memory(resident,workspace,budget).is_err() || reservation>pool {
            return Err(format!("text service memory refusal: CONTEXT={context}; budget={budget}; resident={resident}; shared_workspace={workspace}; session_pool={pool}; full_session={reservation}; reduce CONTEXT"));
        }
        // Kept prompts never take the room of one full-size chat: the engine's byte cap, and the cache's byte bound below
        // is the same number, so the cache gives back its least recently used prompt before the engine would refuse one.
        let kept_cap=if keep {engine.keep_within(pool)} else {0};
        // Startup ends here on every rank: the weights are resident and the ranks agree on memory. Rank 0 opens its port right after.
        startup::done();
        if !engine.controller() {return engine.worker();}
        if let Some(dir)=gate.as_deref() {
            let result=boundary_gate::run(&engine,dir,context,budget,|| (backend.accounted_bytes(),backend.peak_accounted_bytes(),spark_gpu_probe::driver::mhc_fused_launches()));
            let stopped=engine.shutdown();return result.and(stopped);
        }
        // DEFAULT DRAFTING: a trial makes its first shadow where the answer leaves its think block (row_split/trial.rs).
        // The tokenizer's own encode reads control markers; the service's encode of request text does not.
        if let Ok(ids)=std::ops::Deref::deref(&tokenizer).encode("</think>") && let [marker]=ids.as_slice() {engine.set_answer_marker(*marker);eprintln!("TEXT TRIAL MARKER rank={rank} token={marker}");}
        let mut limits=spark_tp::row_split::limits(context,setting("FLASH_SERVE_OUTPUT_TOKENS",output_setting)?.unwrap_or(context.min(8192)))?;
        limits.next_mtp_depth=Some(engine.predictor_depth());limits.api_key=auth;
        limits.metadata=Some(spark_serve::metadata::Metadata::resident_text(&identity,world,wire_topology));
        limits.identity=Some(identity);
        limits.validate()?;
        // The request body allowance grows with the context; the tokenizer's default 1 MiB input cap would refuse a long prompt that fits.
        tokenizer.set_limits(limits.tokenizer_limits())?;
        // With --no-prompt-keep the cache holds nothing (zero bytes), as before the cache existed. Without it (the
        // default) the cache keeps up to as many prompts as each rank can hold, inside the session pool: live chats reserve first and the cache gives bytes
        // back before a request waits (`rebalance_within`). A kept prompt is charged all it holds (`retained_bytes`):
        // it holds no expanded rows, a chat that starts from it builds them again.
        let cache=spark_serve::cache::CacheLimits {entries:if keep {spark_tp::row_split::KEPT_PROMPTS} else {1},retained_bytes:kept_cap,tokens_per_entry:context,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
        // DEFAULT DRAFTING: with a predictor of at least TRIAL_DEPTH loaded, a request that names no draft mode drafts
        // as a strict one while its chat is alone (row_split.rs configure_request); "plain" and mtp_depth 0 turn it off.
        let default_depth=if engine.predictor_depth()>=spark_tp::row_split::TRIAL_DEPTH {spark_tp::row_split::TRIAL_DEPTH} else {0};
        let schedule=spark_serve::scheduler::Config {active:spark_tp::row_split::LIVE_SESSIONS,connections:32,prefill_chunk:spark_tp::bounded_decoder::PROMPT_ROWS,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,
            draft:if default_depth>0 {spark_serve::protocol::DraftMode::Strict} else {spark_serve::protocol::DraftMode::Plain},mtp_depth:default_depth,
            draft_global_bank:false,prefix_cache:keep,sharing:spark_serve::scheduler::Sharing {private_counts:keep,..Default::default()}};
        mailbox.require_success()?;
        let listener=TcpListener::bind(bind).map_err(|e|e.to_string())?;let stop=AtomicBool::new(false);
        let name=root.file_name().ok_or("checkpoint needs a directory name")?.to_string_lossy().to_ascii_lowercase();
        // The cache's room for the start line: its byte cap and the longest prompt whose PLANNED charge fits under it (the
        // engine's loose invoice of a kept prompt and the cache's own notes; the cache books a real prompt by the bytes its
        // state holds on the device), and a notice when that is small.
        let largest=if keep {super::settings::text_largest_kept_prompt(context,kept_cap,|tokens|spark_serve::cache::publication_metadata_bytes(tokens,engine.vocab_size(),engine.kept_prompt_bytes(tokens),0,0))} else {0};
        if let Some(notice)=super::settings::text_keep_room_notice(keep,context,kept_cap,largest) {eprintln!("TEXT PROMPT KEEP rank={rank} {notice}");}
        eprintln!("text_service qualification=unqualified diagnostic_option=--diagnostic-unqualified-text ready bind={bind} rank={rank} world={world} path=tensor_split context={context} resident_bytes={resident} session_bytes={reservation} shared_workspace_bytes={workspace} session_pool_bytes={pool} max_chats=8 {} cache={keep} {}",super::settings::text_drafting_words(default_depth,engine.predictor_depth()),super::settings::text_keep_room_words(kept_cap,largest));
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
