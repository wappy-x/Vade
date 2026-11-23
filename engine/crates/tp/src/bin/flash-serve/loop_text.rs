//! The start of the text service of the looped block family on one Spark, behind `--diagnostic-unqualified-text`.
//!
//! The library holds the service and its start for any backend (`spark_tp::loop_decoder::service::start`). This
//! file is the Linux half of a start: the memory rule, and for the device run the GPU context, the load of the
//! weights and the call. With `--diagnostic-host-arithmetic` it starts the host run of the library and opens no GPU.
//! The steps of a device start, in the words of the startup lines: `header_admission`, `tokenizer_identity`,
//! `memory`, `memory_admission`, `gpu_context_and_kernel_image`, `weights`, `host_weights`, then the ready line.
use engine_model::plan::{description::LoopTextDescription,loop_load};
use spark_tp::{Result,bounded_decoder::Device,loop_decoder::{operands::{device::DeviceStore,load,weights::LoadedWeights},service::{STEP_LIMITS,admit,start}},startup};
use std::{net::TcpListener,path::Path,sync::atomic::AtomicBool,time::Instant};

/// The memory margin of the vector pool: the pool keeps no spare that would leave less than this free.
const VECTOR_POOL_MARGIN:usize=1<<30;
/// The rule of the grid of a launch whose extent follows the context: a straight plane holds whole buckets.
const BUCKET:u64=spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;

/// Start the service when `root` holds a model of this family and the start gives the diagnostic option.
/// `None`: another model, or no diagnostic option; the caller goes on as it did before this file.
/// `output` and `memory` are the caller's settings of the output cap and of the memory ceiling, as text.
pub fn run(args:&[String],root:&Path,output:Option<String>,memory:Option<String>,started:Instant)->Option<Result<()>> {
    if !args.get(9..).unwrap_or_default().iter().any(|option|option==start::DIAGNOSTIC) {return None;}
    let config=engine_model::safetensors::read_bounded(&root.join("config.json"),engine_model::shapes::CONFIG_BYTES).ok()?;
    LoopTextDescription::from_config(&config).ok()?;
    Some(serve(args,root,output,memory,started))
}
/// The budget under the caller's memory ceiling, when it gives one.
fn ceiling(budget:usize,memory:Option<String>)->Result<usize> {
    Ok(match memory {Some(text)=>budget.min(text.parse::<usize>().ok().filter(|&bytes|bytes>0).ok_or("SEQPAR_MEMORY_BYTES must be a positive byte count")?),None=>budget})
}
fn serve(args:&[String],root:&Path,output:Option<String>,memory:Option<String>,started:Instant)->Result<()> {
    let settings=start::settings(args,output)?;
    startup::begin(settings.rank,started,"header_admission");
    if settings.options.host {
        // The host run opens no GPU. Its budget is the memory rule on the free host memory.
        startup::step("memory");
        let free=spark_gpu_probe::driver::unified_available_bytes(usize::MAX)?;
        let budget=ceiling(free.saturating_sub((free/10).max(2<<30)),memory)?;
        let listener=TcpListener::bind(settings.bind).map_err(|e|e.to_string())?;
        return start::serve_host(root,&settings,budget as u64,BUCKET,listener,&AtomicBool::new(false));
    }
    let checkpoint=loop_load::read_folder(root).map_err(|e|format!("This engine cannot load this model: {e}."))?;
    let d=&checkpoint.description;
    startup::step("tokenizer_identity");
    let text=start::text(root,d)?;
    // The kernel image of the block arms, which every step runs: a build without it stops here, before the GPU opens.
    startup::step("kernel_image");
    start::need_kernel_image(spark_gpu_probe::inkling_ptx().is_some())?;
    startup::step("memory");
    let (_,total)=spark_gpu_probe::driver::device_memory_info(0)?;
    let free=spark_gpu_probe::driver::unified_available_bytes(total)?;
    // The memory rule of the text service: the free bytes less the larger of one tenth and 2 GiB.
    let budget=ceiling(free.saturating_sub((free/10).max(2<<30)),memory)?;
    startup::step("memory_admission");
    let admitted=admit(d,&checkpoint.load.bytes,settings.context,budget as u64,BUCKET,STEP_LIMITS)?;
    eprintln!("LOOP TEXT ADMISSION context={} available_bytes={free} budget_bytes={budget} weights_bytes={} chat_bytes={} ledger_bytes={} session_pool_bytes={} largest_context={} step_rows={}",
        settings.context,admitted.ledger.weights,admitted.ledger.chat(),admitted.ledger.total,admitted.room()?,admitted.largest_context,admitted.limits.step_rows);
    let ptx=spark_gpu_probe::layer_ptx()?;
    let workers=std::thread::available_parallelism().map_or(4,|n|n.get()).min(16);
    startup::step("gpu_context_and_kernel_image");
    spark_gpu_probe::driver::with_backend(0,ptx,budget,|backend| {
        // The kernel image of the block arms, loaded before the weights, so the first chat does not pay for it.
        startup::step("kernel_image_load");
        backend.inkling_preload()?;
        let before=backend.allocated_bytes();
        startup::step("weights");
        let mut store=DeviceStore::new(backend);
        let loaded=load::load(&checkpoint,root,&mut store,workers)?;
        backend.synchronize()?;
        let resident=backend.allocated_bytes()-before;
        // The ledger admitted the context before the load. The device must hold what the ledger counted.
        if resident as u64!=admitted.ledger.weights {return Err(format!("the device holds {resident} bytes of weights; the ledger counted {}",admitted.ledger.weights));}
        startup::step("host_weights");
        let weights=LoadedWeights::new(&checkpoint,&load::Files::open(&checkpoint,root)?,&loaded.operands)?;
        eprintln!("LOOP TEXT LOAD tensors={} units={} fingerprint={} device_weights_bytes={resident} host_weight_bytes={} allocate_seconds={:.3} fill_seconds={:.3} seconds={:.3}",
            loaded.report.tensors,loaded.report.units,loaded.fingerprint(),weights.host_bytes(),loaded.report.allocate_seconds,loaded.report.fill_seconds,started.elapsed().as_secs_f64());
        // Freed device vectors wait, by exact length, for the next vector of that length (the backend's vector pool):
        // a token step then allocates almost nothing. Switched on after the load, so the check of the weight bytes
        // above counts no spare. The pool gives its spares back when the budget needs them.
        backend.configure_vector_pool(VECTOR_POOL_MARGIN);
        let compute=Device::new(backend);
        let listener=TcpListener::bind(settings.bind).map_err(|e|e.to_string())?;
        let stop=AtomicBool::new(false);
        let served=start::serve(&compute,&weights,d,text,&admitted,&settings,listener,root,"device",&stop);
        let drained=backend.synchronize();
        served.and(drained)
    })
}
