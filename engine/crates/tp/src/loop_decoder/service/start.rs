//! The start of the text service of the looped block family, for any backend: the options, what the start reads
//! from the model folder beside the weights, and the serving loop.
//!
//! The caller opens its backend, loads the weights and admits the context (`super::admit`). Then it calls `serve`.
//! So the driver's host run on a CPU and its device run start the same service: only the backend and the owner of
//! the weights differ.
//!
//! The service is an unqualified diagnostic service, as the other text service was at its start: it needs the
//! explicit diagnostic option and a loopback HTTP bind. It has no transport between Sparks, so it runs on a rank
//! that has no neighbor.
use super::{Admitted,Service,admit,limits};
use crate::{Result,bounded_decoder::{Compute,Host},loop_decoder::{host_weights::HostWeights,mix::Weights}};
use engine_model::plan::{description::LoopTextDescription,layer::Topology,loop_load::LoadBytes};
use spark_gpu_probe::tp_route::{Links,Side};
use spark_serve::{Tokenizer,contract::Identity,tokenizer::ModelTokenizer};
use std::{net::{SocketAddr,TcpListener},path::{Path,PathBuf},sync::atomic::AtomicBool,time::Instant};
#[cfg(test)] mod tests;

/// The option that a start must give: this service is not qualified.
pub const DIAGNOSTIC:&str=engine_model::plan::inkling_index::PREVIEW_OPTION;
/// The option that selects the driver's host run: 32-bit host arithmetic on the CPU, and no GPU.
pub const HOST_ARITHMETIC:&str="--diagnostic-host-arithmetic";
/// What the service takes after REQUESTS. Its refusals print this list.
pub const OPTIONS:&str="this text service takes --diagnostic-unqualified-text, --diagnostic-host-arithmetic (the decoder on the CPU, no GPU), --mtp DEPTH, --no-prompt-keep and --api-key-file PATH, each one time";
/// The places of a start before its options.
pub const PLACES:&str="this text service needs RANK WORLD LISTEN NEXT PTX MODEL CONTEXT BIND REQUESTS";

fn need(ok:bool,why:impl FnOnce()->String)->Result<()> {if ok {Ok(())} else {Err(why())}}
/// The line of a device start whose build has no Inkling kernel image. Every step of this service runs the block arms,
/// which live in that image, so without it the first chat would fail after a start that looked well.
pub const NO_KERNEL_IMAGE:&str="the start stops: this build has no Inkling kernel image, and every chat of the loop text service needs it. Run `cargo kernels-inkling`, then build again";
/// A device start goes on only when the build holds the kernel image that the plan needs (`present`: the answer of
/// the backend's own image check). Else it stops with one line, before the weights are loaded.
pub fn need_kernel_image(present:bool)->Result<()> {need(present,||NO_KERNEL_IMAGE.to_owned())}
#[derive(Debug,Clone,PartialEq,Eq,Default)]
pub struct Options {pub diagnostic:bool,pub host:bool,pub key:Option<PathBuf>,pub no_prompt_keep:bool,pub depth:usize}
/// Read the options after REQUESTS. Unsupported and duplicate options are refused with the list.
pub fn options(args:&[String])->Result<Options> {
    let (mut options,mut mtp)=(Options::default(),false);let mut args=args.iter();
    while let Some(option)=args.next() {match option.as_str() {
        DIAGNOSTIC if !options.diagnostic=>options.diagnostic=true,
        HOST_ARITHMETIC if !options.host=>options.host=true,
        "--no-prompt-keep" if !options.no_prompt_keep=>options.no_prompt_keep=true,
        "--api-key-file" if options.key.is_none()=>options.key=Some(PathBuf::from(args.next().ok_or("--api-key-file needs a path")?)),
        "--mtp" if !mtp=>{
            options.depth=args.next().ok_or("--mtp needs a depth")?.parse().map_err(|_|format!("--mtp needs a nonnegative depth; {OPTIONS}"))?;mtp=true;
        },
        _=>return Err(format!("unsupported text service option: {option}; {OPTIONS}")),
    }}
    need(!options.host || options.depth==0,||"drafting requires the GPU backend".into())?;
    Ok(options)
}
/// The settings of one start: the places CONTEXT, BIND and REQUESTS, the options, and the output cap.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Settings {
    pub rank:usize,pub world:usize,
    /// Tokens of one chat, prompt and answer together.
    pub context:usize,
    pub bind:SocketAddr,
    /// Requests to answer before the service stops. Zero: it goes on until it is stopped. A health request counts.
    pub requests:usize,
    /// The most tokens of one answer.
    pub output:usize,
    pub options:Options,
}
/// Read the places and the options of a start. `output` is the output cap setting of the caller, as text.
/// A start without the diagnostic option, with a bind that is not loopback, or on a rank with a neighbor is refused.
pub fn settings(args:&[String],output:Option<String>)->Result<Settings> {
    need(args.len()>=9,||format!("{PLACES}; {OPTIONS}"))?;
    let number=|place:usize,name:&str|args[place].parse::<usize>().map_err(|_|format!("{name} must be a whole number, got {}",args[place]));
    let (rank,world,context,requests)=(number(0,"RANK")?,number(1,"WORLD")?,number(6,"CONTEXT")?,number(8,"REQUESTS")?);
    let options=options(&args[9..])?;
    need(options.diagnostic,||format!("text service refused: this engine has not qualified this model family; it starts only as an unqualified diagnostic service on a loopback bind: give {DIAGNOSTIC}"))?;
    let bind=args[7].parse::<SocketAddr>().map_err(|e|format!("BIND: {e}"))?;
    need(bind.ip().is_loopback(),||format!("{DIAGNOSTIC} requires a loopback HTTP bind"))?;
    alone(rank,world)?;
    need(context>0 && context<=u32::MAX as usize,||"CONTEXT must be a positive token count inside the token position range".to_owned())?;
    let output=match output {
        Some(value)=>value.parse::<usize>().ok().filter(|&value|value>0).ok_or("FLASH_SERVE_OUTPUT_TOKENS must be a positive token count")?,
        None=>context.min(8192),
    };
    Ok(Settings {rank,world,context,bind,requests,output,options})
}
/// The service has no transport between Sparks. The layout of the start says if this place is one of its places
/// (the check of the model crate) and if it has a neighbor (the links of the layout): a place with one cannot be
/// served.
pub fn alone(rank:usize,world:usize)->Result<()> {
    let place=Topology {rank:rank.try_into().map_err(|_|"RANK exceeds the layout range")?,parts:world.try_into().map_err(|_|"WORLD exceeds the layout range")?};
    place.validate().map_err(|e|format!("RANK {rank} of WORLD {world}: {e}"))?;
    let links=Links::new(world,false)?;
    let neighbors=[Side::Left,Side::Right].into_iter().filter_map(|side|links.neighbor(rank,side)).count();
    need(neighbors==0,||format!("this text service has no transport between Sparks: RANK {rank} of WORLD {world} has {neighbors} neighbor links; start it as RANK 0 of WORLD 1"))
}
/// What a start reads from the model folder beside the weights.
pub struct Text {
    /// The request reader and the reply reader of the chat template of the folder.
    pub tokenizer:ModelTokenizer,
    pub identity:Identity,
    /// The stop set: the EOS list of the checkpoint.
    pub eos:Vec<u32>,
}
/// Read the tokenizer, the chat template and the identity files of `root`, and take the stop set.
/// The stop set is the EOS list of the checkpoint. The list of the config, the list of the generation metadata and
/// the tokenizer must agree, or the start is refused.
pub fn text(root:&Path,description:&LoopTextDescription)->Result<Text> {
    // The tokenizer and the identity hashes read different files, so they run at the same time.
    let (tokenizer,identity)=std::thread::scope(|scope| {
        let tokenizer=scope.spawn(||ModelTokenizer::from_directory(root));
        let identity=Identity::load(root);
        (tokenizer.join().map_err(|_|"tokenizer reader stopped".to_owned()).and_then(|loaded|loaded),identity)
    });
    let (tokenizer,identity)=(tokenizer?,identity?);
    // The model writes its answer as messages of control tokens. The folder's template must select the reader of them.
    need(tokenizer.token_channels(),||"text service refused: the chat template of the model folder selects no reply reader for messages of control tokens".to_owned())?;
    let eos=tokenizer.checkpoint_eos_token_ids()?.ok_or("text service requires checkpoint EOS metadata")?.to_vec();
    let declared=&description.eos_tokens;
    need(!eos.is_empty() && eos.len()==declared.len() && declared.iter().all(|&id|eos.iter().any(|&stop|u64::from(stop)==id)) && eos.iter().all(|&id|tokenizer.has_token(id)),
        ||"service and tokenizer EOS sets differ".to_owned())?;
    Ok(Text {tokenizer,identity,eos})
}
/// Serve chats on `listener` (the caller binds `Settings::bind`) until `stop` is set or `requests` requests are answered.
/// `backend` is the word of the ready line and of the health record for the arithmetic that runs: `host` or `device`.
/// The ready line is the last line before the port answers.
#[allow(clippy::too_many_arguments)] // one call from each start; a struct of these borrows would only rename them.
pub fn serve<C:Compute,W:Weights<C>>(compute:&C,weights:&W,description:&LoopTextDescription,text:Text,admitted:&Admitted,settings:&Settings,listener:TcpListener,root:&Path,backend:&str,stop:&AtomicBool)->Result<()> {
    serve_with_draft(compute,weights,description,text,admitted,settings,listener,root,backend,stop,None)
}
#[allow(clippy::too_many_arguments)]
pub fn serve_with_draft<C:Compute,W:Weights<C>>(compute:&C,weights:&W,description:&LoopTextDescription,text:Text,admitted:&Admitted,settings:&Settings,listener:TcpListener,root:&Path,backend:&str,stop:&AtomicBool,model:Option<&dyn super::mtp::Model<C,W>>)->Result<()> {
    let Text {tokenizer,identity,eos}=text;
    let auth=spark_serve::auth::ApiKey::load(settings.options.key.as_deref(),false)?;
    need(admitted.context==settings.context,||"the admitted context differs from the context of the start".to_owned())?;
    let mut seats=admitted.seats()?;
    if let Some(model)=model {
        let cfg=model.config();
        need(settings.options.depth<=cfg.gamma() && description.declared_draft.as_ref().is_some_and(|d|d.taps.iter().copied().eq(cfg.target_layer_ids.iter().map(|&n|n as u64))),||"draft depth or taps differ from the checkpoint".into())?;
        let kv=settings.context.checked_mul(cfg.kv_width()).and_then(|n|n.checked_mul(8*cfg.num_layers)).ok_or("draft chat size overflow")?;
        // Contended chats retire drafting, so reserve a bounded draft KV pool (up to 8 chats) off room
        // instead of charging full draft KV to every chat slot.
        let pool=kv.checked_mul(seats.most.min(8)).ok_or("draft reservation overflow")?;
        let charge=if seats.room>=pool.saturating_add(seats.chat) {pool} else {kv};
        seats.room=seats.room.checked_sub(charge).ok_or("text service memory refusal: no room for draft KV")?;
    }
    let mut service=Service::new(compute,weights,description,settings.context,eos,seats)?.with_prompt_keep(!settings.options.no_prompt_keep);
    service.model=model;service.depth=settings.options.depth;
    // Keep spares for one step plus pooled session state so repeat cohorts reuse arenas without cuMemAlloc stalls.
    let spares=service.step_bytes().max(service.state_bytes().saturating_mul(service.chats().min(32)).min(service.pool()));
    compute.step_spares(spares);
    let mut limits=limits(settings.context,settings.output)?;
    limits.next_mtp_depth=Some(service.depth);limits.api_key=auth;limits.metadata=Some(spark_serve::metadata::Metadata::loop_text(&identity,backend));limits.identity=Some(identity);
    limits.validate()?;
    let (cache,config)=service.schedule();
    let name=root.file_name().ok_or("checkpoint needs a directory name")?.to_string_lossy().to_ascii_lowercase();
    eprintln!("LOOP TEXT ROWS step_rows={} key_group_bytes={} room_bytes={} step_bytes={} session_pool_bytes={} chats={} chat_bytes={} ledger_chat_bytes={} chat_state_bytes={}",
        service.limits().step_rows,service.limits().key_bytes,admitted.room()?,service.step_bytes(),service.pool(),service.chats(),service.chat_bytes(),admitted.ledger.chat(),service.state_bytes());
    crate::startup::done();
    eprintln!("text_service qualification=unqualified diagnostic_option={DIAGNOSTIC} ready bind={} rank={} world={} path=loop_driver backend={backend} context={} output={} resident_bytes={} session_bytes={} session_pool_bytes={} max_chats={} mtp_depth={} cache={}",
        listener.local_addr().map_err(|e|e.to_string())?,settings.rank,settings.world,settings.context,settings.output,admitted.ledger.weights,service.chat_bytes(),service.pool(),service.chats(),service.depth,config.prefix_cache);
    spark_serve::scheduler::serve_requests(listener,&service,Some(&tokenizer),&limits,&cache,&config,stop,&name,settings.requests)
}
/// The most bytes of decoded expert matrices that the host run keeps: a quarter of its budget, and at most 24 GiB.
/// A host resource limit: it changes no word.
pub fn host_expert_cache(budget:u64)->usize {usize::try_from((budget/4).min(24<<30)).unwrap_or(usize::MAX)}
/// The host run of a start: the weights of the model folder as host words (`HostWeights`) and the driver on the CPU
/// with 32-bit host arithmetic. No GPU is opened. `budget` is the host memory that the run may take. The dense
/// weights, the limit of the expert cache and the state of one chat are admitted against it: the ready line gives
/// the first two as its resident bytes. `bucket` is the row rule of the state planes, as for the device.
pub fn serve_host(root:&Path,settings:&Settings,budget:u64,bucket:u64,listener:TcpListener,stop:&AtomicBool)->Result<()> {
    need(settings.options.host,||format!("the host run of this text service needs {HOST_ARITHMETIC}"))?;
    crate::startup::step("host_weights");
    let (began,cache)=(Instant::now(),host_expert_cache(budget));
    let (description,weights)=HostWeights::open(root,cache)?;
    let dense=(weights.counts()?.dense_words as u64).checked_mul(4).ok_or("the dense weights exceed the byte range")?;
    let loaded=began.elapsed().as_secs_f64();
    crate::startup::step("tokenizer_identity");
    let text=text(root,&description)?;
    crate::startup::step("memory_admission");
    let load=LoadBytes {device:dense.checked_add(cache as u64).ok_or("the host weights exceed the byte range")?,..Default::default()};
    let admitted=admit(&description,&load,settings.context,budget,bucket,super::STEP_LIMITS)?;
    eprintln!("LOOP TEXT HOST context={} budget_bytes={budget} dense_weight_bytes={dense} expert_cache_limit_bytes={cache} chat_bytes={} largest_context={} load_seconds={loaded:.3}",
        settings.context,admitted.ledger.chat(),admitted.largest_context);
    let served=serve(&Host,&weights,&description,text,&admitted,settings,listener,root,"host",stop);
    if let Ok(counts)=weights.counts() {eprintln!("LOOP TEXT HOST END experts_decoded={} cache_hits={} held={} held_bytes={}",counts.decoded,counts.hits,counts.held,counts.held_bytes);}
    served
}
